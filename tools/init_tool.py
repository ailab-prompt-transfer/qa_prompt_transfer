import logging
import torch
from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function
from torch import nn

import os

logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, mode, *args, **params):
    result = {}
    logger.info("Begin to initialize dataset and formatter...")
    if mode == "test":
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)
    elif mode == "train" or mode == "valid":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    else:
        print("Don't need to load data")

    logger.info("Begin to initialize models...")
    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    if len(gpu_list) > 0:
        if params["local_rank"] < 0:
            model = model.cuda()
        else:
            ###
            # muti machines
            model = model.to(gpu_list[params["local_rank"]])
        try:
            # muti machines
            model = nn.parallel.DistributedDataParallel(model, device_ids=[params["local_rank"]], output_device=params["local_rank"], find_unused_parameters=True)
        except Exception as e:
            logger.warning("No init_multi_gpu implemented in the model, use single gpu instead.")

    # print(params) #{'local_rank': -1, 'prompt_emb_output': True}
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0

    if params["args"].checkpoint != None and mode == "train":
        model_size = config.get("model", "model_size")
        if "base" in model_size:
            load_dir = "T5ForMaskedLM/PromptT5_init_params/pytorch_model.bin"
        elif "small" in model_size:
            load_dir = "T5SmallForMaskedLM/PromptT5_init_params/pytorch_model.bin"
        else:
            logger.error("There is no model_size")
            raise NotImplementedError

        if os.path.exists(load_dir):
            parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
        else:
            print("Not exist:", load_dir)
            exit()

        for key in list(parameters):
            parameters["encoder." + key] = parameters.pop(key)

        if hasattr(model, "module"):
            model.module.load_state_dict(parameters)
        else:
            model.load_state_dict(parameters)

        load_checkpoint = params["args"].checkpoint

        print()
        print("#############################################################")
        print("Best Checkpoint : ", load_checkpoint)
        print("#############################################################")
        print()
        prompt_parameters = torch.load(load_checkpoint, map_location=lambda storage, loc: storage)

        model.encoder.prompt_embeddings.weight.data = prompt_parameters["model"]
        model.encoder.encoder.prompt_tokens.weight.data = prompt_parameters["model"]
        model.encoder.decoder.prompt_tokens.weight.data = prompt_parameters["model"]

        if torch.cuda.is_available() and mode == "train":
            model.cuda()
        else:
            pass

        if config.get("train", "optimizer") == prompt_parameters["optimizer_name"]:
            optimizer.load_state_dict(prompt_parameters["optimizer"])
        trained_epoch = 0
        global_step = 0
    else:
        pass

    if mode == "valid" or mode == "Valid" or mode == "test" or mode == "Test":
        print("=========================")
        print("params")
        print(params)
        print(f'param[args].replacing_prompt: {params["args"].replacing_prompt}')
        print("=========================")

        if "Random" in params["args"].replacing_prompt or "random" in params["args"].replacing_prompt and params["args"].replacing_prompt != "randomPromptRobertaLarge":
            print("=========================")
            print("Using random prompt emb")
            print("=========================")
            config_name = params["args"].config.split("/")[1].split(".")[0]
            if "Large" in config_name or "large" in config_name:
                prompt_emb = torch.rand(config.getint("prompt", "prompt_num"), 1024).to("cuda")
            if "Small" in config_name:
                prompt_emb = torch.rand(config.getint("prompt", "prompt_num"), 512).to("cuda")
            else:  # Base
                prompt_emb = torch.rand(config.getint("prompt", "prompt_num"), 768).to("cuda")
        elif params["args"].replacing_prompt != None:
            print("=========================")
            print(f"Using {params['args'].replacing_prompt} prompt emb")
            print("=========================")

            load_task_prompt_dir = params["args"].replacing_prompt
            prompt_emb = torch.load(load_task_prompt_dir, map_location=lambda storage, loc: storage)
        else:
            logger.error("There is no replacing_prompt")
            raise NotImplementedError

        if prompt_emb != None:
            try:
                prompt_emb = torch.nn.Parameter(prompt_emb["model"]).to("cuda")
            except:
                prompt_emb = torch.nn.Parameter(prompt_emb).to("cuda")

            if "T5" in params["args"].config:
                model.encoder.prompt_embeddings.weight.data = prompt_emb
                model.encoder.encoder.prompt_tokens.weight.data = prompt_emb
                model.encoder.decoder.prompt_tokens.weight.data = prompt_emb
            else:
                print("Wrong!!!")
                exit()
        else:
            print("=========================")
            print("Using original prompt emb")
            print("=========================")
            pass
    else:
        ####Train####
        print("Mode: Train")
        pass

    result["model"] = model
    result["optimizer"] = optimizer
    result["trained_epoch"] = trained_epoch
    result["output_function"] = init_output_function(config)
    result["global_step"] = global_step

    logger.info("Initialize done.")

    return result
