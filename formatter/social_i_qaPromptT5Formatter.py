from transformers import AutoTokenizer
from transformers import T5Tokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

"""
    Cosmos QA ) Open-book QA
"""


class social_i_qaPromptT5Formatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        # self.max_len = config.getint("train", "max_len")
        self.target_len = config.getint("train", "target_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
        # self.prompt_num = config.getint("prompt", "prompt_num")
        self.mode = mode

        self.model_name = config.get("model", "model_base")
        if "T5" in self.model_name:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        else:
            print("Have no matching in the formatter")
            exit()

        self.prompt_prefix = [-(i + 1) for i in range(self.prompt_len)]

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        text_label = []
        # max_len = self.max_len + 2 + self.prompt_num#+ self.prompt_len * 1 + 4

        max_len = self.tokenizer.model_max_length - self.prompt_len  # 412
        for ins in data:
            choice_token = "<extra_id_0>"
            text = ["question:", ins["question"], "choice:", choice_token, ins["answerA"], choice_token, ins["answerB"], choice_token, ins["answerC"], "context:", ins["context"]]

            sent = self.tokenizer.encode(" ".join(text), add_special_tokens=True, max_length=max_len)

            tokens = self.prompt_prefix + sent

            # padding & attention_masks
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.tokenizer.model_max_length - len(tokens))
            mask.append([1] * len(tokens) + [0] * (self.tokenizer.model_max_length - len(tokens)))

            # lable이 1부터 시작함
            if ins["label"] == 1:
                ins["label"] = ins["answerA"]
            elif ins["label"] == 2:
                ins["label"] = ins["answerB"]
            else:
                ins["label"] = ins["answerC"]

            # Update
            if mode == "train":
                # Target
                target = self.tokenizer(ins["label"], add_special_tokens=True, max_length=self.target_len, truncation=True)["input_ids"]

                # padding
                target = target + [-100] * (self.target_len - len(target))

                label.append(target)
                text_label.append(ins["label"])
            else:
                label.append(ins["label"])

            inputx.append(tokens)

        if mode == "train":
            ret = {
                "inputx": torch.tensor(inputx, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float),
                "label": torch.tensor(label, dtype=torch.long),
                "text_label": text_label,
            }
        else:  # test / valid
            ret = {
                "inputx": torch.tensor(inputx, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float),
                "label": label,
            }

        return ret
