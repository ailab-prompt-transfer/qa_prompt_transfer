from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter


class restaurantPromptRobertaFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
        self.prompt_num = config.getint("prompt", "prompt_num")
        self.target_len = config.getint("train", "target_len")
        self.mode = mode
        ##########
        self.model_name = config.get("model", "model_base")
        if "Roberta" in self.model_name:
            # self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")
        elif "Bert" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            print("Have no matching in the formatter")
            exit()

        self.prompt_prefix = [-(i + 1) for i in range(self.prompt_len)]

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        text_label = []
        max_len = self.max_len + 3 + self.prompt_num  # + self.prompt_len * 1 + 4
        for ins in data:
            sent = self.tokenizer.encode(ins["sent"], add_special_tokens=False)

            if len(sent) >= self.max_len:
                sent = sent[: self.max_len]
            tokens = self.prompt_prefix + [self.tokenizer.cls_token_id] + sent + [self.tokenizer.sep_token_id]

            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            if mode == "train":
                label.append(tokens)
                text_label.append(tokens)
            else:
                label.append(ins["label"])

            inputx.append(tokens)

        if mode == "train":
            ret = {
                "inputx": torch.tensor(inputx, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float),
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            ret = {
                "inputx": torch.tensor(inputx, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float),
                "label": label,
            }

        return ret
