import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset


class nq_openDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset("nq_open")

        # train : valid = 9:1 로 split
        TRAIN_LEN = len(self.data["train"])
        NQ_TRAIN_SPLIT_END = int(TRAIN_LEN * 0.9)

        if self.mode == "train" or self.mode == "valid":
            self.data = self.data["train"]
        else:  # test
            self.data = self.data["validation"]

        data = [row for row in self.data]
        if self.mode == "train":
            data = data[:NQ_TRAIN_SPLIT_END]
        elif self.mode == "valid":
            data = data[NQ_TRAIN_SPLIT_END:]

        if self.mode == "train":
            self.data = [{"question": ins["question"].strip(), "label": ins["answer"][0].strip()} for ins in data]
        else:  # multi-answer
            self.data = [{"question": ins["question"].strip(), "label": [x.strip() for x in ins["answer"]]} for ins in data]

        # debug
        print(f"\n==============================")
        print(f"mode: {mode}, size: {len(self.data)}")
        print(f"train_len : {TRAIN_LEN}, NQ_TRAIN_SPLIT_END : {NQ_TRAIN_SPLIT_END}")
        print(f"==============================\n")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
