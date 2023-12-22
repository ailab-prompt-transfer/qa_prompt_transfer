from torch.utils.data import Dataset
from datasets import load_dataset


class cosmos_qaclosedDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset("cosmos_qa")

        # train : valid = 9:1 split
        TRAIN_LEN = len(self.data["train"])
        COSMOS_TRAIN_SPLIT_END = int(TRAIN_LEN * 0.9)

        if self.mode == "train" or self.mode == "valid":
            self.data = self.data["train"]
        else:  # test
            self.data = self.data["validation"]

        data = [row for row in self.data]
        if self.mode == "train":
            data = data[:COSMOS_TRAIN_SPLIT_END]
        elif self.mode == "valid":
            data = data[COSMOS_TRAIN_SPLIT_END:]

        self.data = [
            {
                "question": ins["question"].strip(),
                "answer0": ins["answer0"].strip(),
                "answer1": ins["answer1"].strip(),
                "answer2": ins["answer2"].strip(),
                "answer3": ins["answer3"].strip(),
                "label": str(ins["label"]),
            }
            for ins in data
        ]

        # debug
        print(f"\n==============================")
        print(f"mode: {mode}, size: {len(self.data)}")
        print(f"==============================\n")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
