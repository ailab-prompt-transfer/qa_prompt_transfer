import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from config_parser import create_config
from tools.test_tool import test


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

logger = logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="specific config file", required=True)
    parser.add_argument("--gpu", "-g", help="gpu id list")
    parser.add_argument("--local_rank", type=int, help="local rank", default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replacing_prompt", type=str, default=None)
    parser.add_argument("--checkpoint", help="checkpoint file path", type=str, default=None)

    args = parser.parse_args()
    configFilePath = args.config
    config = create_config(configFilePath)

    gpu_list = []
    if args.gpu is None:
        use_gpu = False
        exit()
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    set_random_seed(args.seed)

    parameters = init_all(config, gpu_list, None, "test", local_rank=args.local_rank, args=args)
    model = parameters["model"]

    test_result = test(parameters, config, gpu_list)

    cur_dataset_name = config.get("data", "test_dataset_type")
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

    print(f"test_result: {test_result}")
    if "samsum" in config.get("data", "test_dataset_type") or "multi_news" in config.get("data", "test_dataset_type"):
        test_rouge1 = (test_result["rouge1"] / test_result["total_cnt"]) * 100
        test_rouge2 = (test_result["rouge2"] / test_result["total_cnt"]) * 100
        test_rougeL = (test_result["rougeL"] / test_result["total_cnt"]) * 100

        print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (test_rouge1, test_rouge2, test_rougeL))
    elif cur_dataset_name in qa_datasets:
        test_f1 = (test_result["f1"] / test_result["total_cnt"]) * 100
        test_em = (test_result["em"] / test_result["total_cnt"]) * 100

        print(f'Replacing Prompt: \n{args.replacing_prompt}')
        print("EM: %.6f, F1: %.6f" % (test_em, test_f1))
        print()