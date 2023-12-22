from .PromptT5 import PromptT5

model_list = {
    "PromptT5": PromptT5,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
