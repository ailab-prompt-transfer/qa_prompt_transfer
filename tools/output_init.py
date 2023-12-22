from .output_tool import f1_em_output_funtion, basic_output_function, null_output_function, output_function1, acc_output_function, pearson_output_function

output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "out1": output_function1,
    "acc": acc_output_function,
    "pearson": pearson_output_function,
    "f1_em": f1_em_output_funtion,
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
