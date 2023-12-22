import logging
import torch
from torch.autograd import Variable
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return "%2d:%02d" % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 7:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 14:
        s += " "
    s = s + str(step) + " "
    while len(s) < 25:
        s += " "
    s += str(time)
    while len(s) < 40:
        s += " "
    s += str(loss)
    while len(s) < 48:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    if not (end is None):
        print(s, end=end)
    else:
        print(s)


def valid(model, dataset, epoch, writer, config, gpu_list, output_function, mode="valid", local_rank=-1, **kwargs):
    if "args" in kwargs:
        kwargs = kwargs["args"]

    if len(gpu_list) > 1:
        _model = model.module
    else:
        _model = model

    model.eval()
    local_rank = local_rank

    acc_result = None
    total_loss = 0

    step = -1

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

    with torch.no_grad():
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            results = _model(data, config, gpu_list, acc_result, "valid", args=kwargs)

            if cur_dataset_name in qa_datasets:
                acc_result = results["acc_result"]
            else:
                total_loss += float(results["valid_loss"])

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    model.train()

    if not cur_dataset_name in qa_datasets:
        total_loss += float(results["valid_loss"])
        print(f"total_loss:{total_loss}, step:{step} = ({round(total_loss/(step+1),3)})")
        acc_result = {"avg_valid_loss": (total_loss / (step + 1))}

    return acc_result
