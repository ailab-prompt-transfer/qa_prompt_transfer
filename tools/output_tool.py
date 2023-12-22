import json

from .accuracy_tool import gen_micro_macro_result


## output.init에 함수 추가해야 함


def f1_em_output_funtion(data, *args, **parms):
    # f1, em
    if data["f1"] == 0:
        return 0
    else:
        return json.dumps({"f1": round((data["f1"] / data["total_cnt"] * 100), 4), "em": round((data["em"] / data["total_cnt"] * 100), 4)})


def null_output_function(data, config, *args, **params):
    return ""




def acc_output_function(data, config, *args, **params):
    if data["total"] == 0:
        return 0
    else:
        return json.dumps({"acc": round(data["right"] / data["total"], 4)})


def pearson_output_function(data, config, *args, **params):
    if data["batch_num"] == 0:
        return 0
    else:
        return json.dumps({"acc": round(data["total_pearson"] / data["batch_num"], 4)})


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)


def output_function1(data, config, *args, **params):
    if data["pre_num"] != 0 and data["actual_num"] != 0:
        pre = data["right"] / data["pre_num"]
        recall = data["right"] / data["actual_num"]
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
        "precision": round(pre, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    if "labelset" in data and "doc_num" in data and data["doc_num"] != 0:
        metric["ave_len"] = data["labelset"] / data["doc_num"]
    return json.dumps(metric)
