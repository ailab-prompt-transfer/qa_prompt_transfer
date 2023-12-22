from torch.utils.data import Dataset
from datasets import load_dataset


class multircDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset("super_glue", "multirc")

        # train : valid = 9:1 로 split
        TRAIN_LEN = len(self.data["train"])
        MULTIRC_TRAIN_SPLIT_END = int(TRAIN_LEN * 0.9)

        if self.mode == "train" or self.mode == "valid":
            self.data = self.data["train"]
        else:  # test
            self.data = self.data["validation"]

        data = [row for row in self.data]
        if self.mode == "train":
            data = data[:MULTIRC_TRAIN_SPLIT_END]
        elif self.mode == "valid":
            data = data[MULTIRC_TRAIN_SPLIT_END:]

        self.data = [{"context": ins["paragraph"].strip(), "question": ins["question"].strip(), "answer": ins["answer"].strip(), "label": ins["label"]} for ins in data]

        # debug
        print(f"\n==============================")
        print(f"mode: {mode}, size: {len(self.data)}")
        print(f"==============================\n")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
