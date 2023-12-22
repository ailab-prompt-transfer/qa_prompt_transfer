from torch.utils.data import DataLoader
import logging

import formatter as form
from dataset import dataset_list
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)

collate_fn = {}
formatter = {}


def init_formatter(config, task_list, *args, **params):
    if "args" in params:
        for task in task_list:
            # debug
            print(f"formatter: {formatter}")
            formatter[task] = form.init_formatter(config, task, *args, **params)

            def train_collate_fn(data):
                return formatter["train"].process(data, config, "train", args=params)

            def valid_collate_fn(data):
                return formatter["valid"].process(data, config, "valid")

            def test_collate_fn(data):
                return formatter["test"].process(data, config, "test")

            if task == "train":
                collate_fn[task] = train_collate_fn
            elif task == "valid":
                collate_fn[task] = valid_collate_fn
            else:
                collate_fn[task] = test_collate_fn

    else:
        for task in task_list:
            formatter[task] = form.init_formatter(config, task, *args, **params)

            def train_collate_fn(data):
                return formatter["train"].process(data, config, "train", args=params)

            def valid_collate_fn(data):
                return formatter["valid"].process(data, config, "valid")

            def test_collate_fn(data):
                return formatter["test"].process(data, config, "test")

            if task == "train":
                collate_fn[task] = train_collate_fn
            elif task == "valid":
                collate_fn[task] = valid_collate_fn
            else:
                collate_fn[task] = test_collate_fn


def init_one_dataset(config, mode, *args, **params):
    temp_mode = mode

    if mode != "train":
        try:
            config.get("data", "%s_dataset_type" % temp_mode)
        except Exception as e:
            logger.warning("[reader] %s_dataset_type has not been defined in config file, use [dataset] train_dataset_type instead." % temp_mode)
            temp_mode = "train"

    ##########n 아래 if 문은 없어도 될 듯
    #### 이 3가지 타입이 뭔지 모르겠음. task transsfer에는 해당 사항 없어 보임 / 지워도 됨
    if config.get("data", "train_formatter_type") == "projectorPromptRoberta":
        which = "projector"
    elif config.get("data", "train_formatter_type") == "crossPrompt":
        which = "cross"
    elif config.get("data", "train_formatter_type") == "cross_mlmPrompt":  #
        which = "cross"
    ##########
    else:
        which = config.get("data", "%s_dataset_type" % temp_mode)
        print(which)  # config 폴더에 없는 경우 FilenameOnly # dataset이 들어감 # ex) samsum

    if which in dataset_list:
        dataset = dataset_list[which](config, mode, *args, **params)
        batch_size = config.getint("train", "batch_size")
        reader_num = config.getint("train", "reader_num")
        drop_last = False
        if mode in ["valid", "test"]:
            try:
                batch_size = config.getint("eval", "batch_size")
            except Exception as e:
                logger.warning("[eval] batch size has not been defined in config file, use [train] batch_size instead.")

            try:
                shuffle = config.getboolean("eval", "shuffle")
            except Exception as e:
                shuffle = False
                logger.warning("[eval] shuffle has not been defined in config file, use false as default.")
            try:
                reader_num = config.getint("eval", "reader_num")
            except Exception as e:
                logger.warning("[eval] reader num has not been defined in config file, use [train] reader num instead.")

        if config.getboolean("distributed", "use"):
            print(f"distribued!")
            sampler = DistributedSampler(dataset)
        else:
            print(dataset)
            sampler = RandomSampler(dataset)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=reader_num, collate_fn=collate_fn[mode], drop_last=drop_last, sampler=sampler)

        return dataloader
    else:
        logger.error("There is no dataset called %s, check your config." % which)
        raise NotImplementedError


def init_test_dataset(config, *args, **params):
    init_formatter(config, ["test"], *args, **params)
    test_dataset = init_one_dataset(config, "test", *args, **params)

    return test_dataset


def init_dataset(config, *args, **params):
    init_formatter(config, ["train", "valid"], *args, **params)
    train_dataset = init_one_dataset(config, "train", *args, **params)
    valid_dataset = init_one_dataset(config, "valid", *args, **params)

    return train_dataset, valid_dataset


if __name__ == "__main__":
    pass
