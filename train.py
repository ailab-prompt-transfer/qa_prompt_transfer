import argparse
import os
import torch
import logging
import random
import numpy as np

from config_parser import create_config
from tools.init_tool import init_all

from tools.train_tool import train

import wandb
import torch.multiprocessing as mp
import torch.distributed as dist

# 로그 메세지가 기록된 시간 / 로그 레벨 / 로거 이름 / 실제 로그 메세지
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

logger = logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run(rank, args):
    configFilePath = args.config
    config = create_config(configFilePath)
    print(config)

    is_master = rank == 0
    if is_master:
        # print('train.py wandb 주석처리함')
        print(f"Wandb project name: {args.wandb_proj}")
        print(f"Wandb run name : {args.wandb_run}")

        wandb.init(project=args.wandb_proj)
        wandb.run.name = args.wandb_run
        wandb.config.update(args)
        wandb.run.save()

    is_mp = len(args.gpu_list) > 1
    world_size = len(args.gpu_list)
    args.local_rank = rank
    if is_mp:
        config.set("distributed", "gpu_num", world_size)
        config.set("distributed", "use", True)
        config.set("distributed", "backend", "nccl")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        config.set("distributed", "use", False)

    print("=====")
    print("configFilePath : ", configFilePath)
    print("args : ", args)
    print(config.get("data", "train_formatter_type"))
    print("=====")

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and is_mp:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    set_random_seed(args.seed)

    print(args.comment)

    parameters = init_all(config, args.gpu_list, args.checkpoint, "train", local_rank=rank, args=args)
    train(parameters, config, args.gpu_list, args.do_test, rank, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print("1. parser : ", parser)

    parser.add_argument("--config", "-c", help="specific config file", default="config/STSBPromptRoberta.config")
    parser.add_argument("--gpu", "-g", help="gpu id list", default="0")
    parser.add_argument("--checkpoint", help="checkpoint file path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, help="local rank", default=-1)
    parser.add_argument("--do_test", help="do test while training or not", action="store_true", default=False)
    parser.add_argument("--comment", help="checkpoint file path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_initial_model", type=str, default=None)
    parser.add_argument("--prompt_emb_output", type=bool, default=False)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--replacing_prompt", type=str, default=None)
    parser.add_argument("--pre_train_mlm", default=False, action="store_true")
    parser.add_argument("--task_transfer_projector", default=False, action="store_true")
    parser.add_argument("--model_transfer_projector", default=False, action="store_true")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--wandb_proj", type=str, help="wandb Project name")
    parser.add_argument("--wandb_run", type=str, help="wandb Run name")
    parser.add_argument("-p", "--port", type=int, default=29500, help="port")
    parser.add_argument("--source_task", type=str, default=None, help="")
    parser.add_argument("--exp", type=str, default=None, help="")
    parser.add_argument("--early_stop", type=int, default=3)
    args = parser.parse_args()
    print("2. args : ", args)

    # setup GPU
    gpu_list = []
    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))
    args.gpu_list = gpu_list
    print(f"gpu : {gpu_list}")

    if len(args.gpu_list) > 1:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_SOKET_IFNAME"] = "lo"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = f"{args.port}"
        mp.spawn(run, args=(args,), nprocs=len(args.gpu_list), join=True)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = f"{args.port}"
        run(0, args)
