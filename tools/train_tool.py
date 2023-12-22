import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np
from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from reader.reader import init_dataset
from datetime import datetime

import wandb


logger = logging.getLogger(__name__)
now = datetime.now()


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, "module") else model
    ###
    """
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }
    """

    model_to_save = model_to_save.state_dict()
    prompt_emb = model_to_save["encoder.prompt_embeddings.weight"]

    save_params = {
        "model": prompt_emb,
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step,
    }

    try:
        torch.save(save_params, filename)

    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def early_stopping(early_stop_max, early_stop_cnt):
    stop_flag = False

    if early_stop_max <= early_stop_cnt:
        stop_flag = True
        print("*** early stopping!!!!!!!!!!!!!!!!!!!!!!")
    else:
        stop_flag = False

    return stop_flag


def train(parameters, config, gpu_list, do_test=False, local_rank=-1, *args, **kwargs):
    if "args" in kwargs:
        kwargs = kwargs["args"]

    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    print("exp : ", kwargs.exp)
    if kwargs.exp == "exp1_2" or kwargs.exp == "exp_tgt_src":
        # kwargs.exp : exp1_0. exp1_2
        output_path = os.path.join(config.get("output", "model_path"), kwargs.exp, config.get("output", "model_name"), kwargs.source_task)
    else:
        output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    # 파일 시작 날짜, 시간 추가
    output_path = output_path + "/%s/" % (now.strftime("%Y%m%d_%H%M%S"))
    print("output_path : ", output_path)

    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")

    # dir 만들기
    os.makedirs(output_path, exist_ok=True)
    print(f"output_path : {output_path}")

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    postfix = ""

    if trained_epoch == 0:
        shutil.rmtree(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name") + postfix), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name") + postfix), exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name") + postfix), config.get("output", "model_name") + postfix)

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")
    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    best_valid_F1 = 0
    best_valid_loss = 10000

    # 소연 테스트
    early_stop_cnt = 0  # 현재 몇번이나 갱신되지 않았는가?

    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num
        model.train()
        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        performance = 0
        total_loss = 0

        output_info = ""
        step = -1

        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            model.zero_grad()

            if "T5" in config.get("model", "model_base"):
                results = model(data, config, gpu_list, acc_result, "train", args=kwargs, step=step, performance=performance)
                loss, performance = results["loss"], results["performance"]

            total_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % output_time == 0 and local_rank <= 0:
                if "T5" in config.get("model", "model_base"):
                    delta_t = timer() - start_time

                    output_value(
                        current_epoch,
                        "train",
                        "%d/%d" % (step + 1, total_len),
                        "%s/%s" % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)),
                        "\t",
                        "\r",
                        config,
                    )

                    # debug
                    wandb.log({"epoch": current_epoch, "step": (step + 1), "loss": (total_loss / (step + 1))})
                else:
                    output_info = output_function(acc_result, config)

                    delta_t = timer() - start_time

                    output_value(
                        current_epoch,
                        "train",
                        "%d/%d" % (step + 1, total_len),
                        "%s/%s" % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)),
                        output_info,
                        "\r",
                        config,
                    )

                    # debug
                    wandb.log({"epoch": current_epoch, "step": (step + 1), "loss": (total_loss / (step + 1))})

            if "T5" in config.get("model", "model_base") and int(step % 100) == 0:
                print("\t \t \t \t \t \t \t", "Performance:", performance)
                # debug
                if local_rank <= 0:
                    wandb.log({"Performance": performance})

            global_step += 1
            writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)
        try:
            model.module.lower_temp(0.8)
        except:
            pass

        if local_rank <= 0:
            if "T5" in config.get("model", "model_base"):
                pass
            else:
                output_info = output_function(acc_result, config)
                delta_t = timer() - start_time
                output_value(
                    current_epoch,
                    "train",
                    "%d/%d" % (step + 1, total_len),
                    "%s/%s" % (gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                    "%.3lf" % (total_loss / (step + 1)),
                    output_info,
                    None,
                    config,
                )

                # debug
                wandb.log({"epoch": current_epoch, "step": (step + 1), "loss": (total_loss / (step + 1))})

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        if local_rank <= 0:
            # Save prompt
            checkpoint(os.path.join(output_path, "epoch_%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step)

            ####
            writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1), current_epoch)
            ###
            if "T5" in config.get("model", "model_base"):
                pass
            else:
                writer.add_scalar(config.get("output", "model_name") + "_train_epoch_acc", float(acc_result["right"] / acc_result["total"]), current_epoch)
            ###

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid_result = valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function, args=kwargs, local_rank=local_rank)
                print(f"valid_result : {valid_result}")

                if local_rank <= 0:
                    # BEFORE using Valid Loss
                    # valid_loss = valid_result['avg_valid_loss']
                    # print(f'({current_epoch} epoch) Valid Performance: {valid_loss}')

                    # wandb.log({'valid_loss': valid_loss})
                    # if valid_loss < best_valid_loss:
                    #     print(f'({current_epoch} epoch) BEST Valid Performance: {valid_loss} < {best_valid_loss}')

                    #     best_valid_loss = valid_loss
                    #     checkpoint(os.path.join(output_path, "best_%d_epoch.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step)

                    # AFTER using F1 score for validation
                    cur_dataset_name = config.get("data", "train_dataset_type")
                    qa_datasets = [
                        "squad",
                        "squadclosed",
                        "nq_open",
                        "tqa",
                        "tqaclosed",
                        "wq",
                        "duorc",
                        "news_qa",
                        "search_qa",
                        "boolq",
                        "boolqclosed",
                        "multirc",
                        "multircclosed",
                        "cosmos_qa",
                        "cosmos_qaclosed",
                        "social_i_qa",
                        "social_i_qaclosed",
                    ]

                    if cur_dataset_name in qa_datasets:
                        valid_f1 = (valid_result["f1"] / valid_result["total_cnt"]) * 100
                        valid_em = (valid_result["em"] / valid_result["total_cnt"]) * 100
                        wandb.log({"valid_F1": valid_f1, "valid_EM": valid_em})
                        print("Epoch: %d, F1: %.6f, EM: %.6f" % (current_epoch, valid_f1, valid_em))

                        # 소연 test / early stopping
                        if valid_f1 > best_valid_F1:
                            early_stop_cnt = 0
                        else:
                            early_stop_cnt += 1
                            # print(f'현재 early stop : {early_stop_cnt}')

                        if valid_f1 >= best_valid_F1:
                            best_valid_F1 = valid_f1
                            checkpoint(os.path.join(output_path, "best_model.pkl"), model, optimizer, current_epoch, config, global_step)

                if do_test:
                    test_result = valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test", local_rank=local_rank)
                    print(f"test: {test_result}")

            # 소연 test
            if early_stopping(kwargs.early_stop, early_stop_cnt):
                break

        # if local_rank >= 0:
        #     torch.distributed.barrier()
