from torch.utils.data import Dataset
from datasets import load_dataset


class duorcDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset("duorc", "SelfRC")

        if self.mode == "train":
            self.data = self.data["train"]
        elif self.mode == "valid":
            self.data = self.data["validation"]
        else:  # test
            self.data = self.data["test"]

        data = [row for row in self.data]

        if self.mode == "train":
            self.data = [{"context": ins["plot"].strip(), "question": ins["question"].strip(), "label": ins["answers"][0].strip()} for ins in data if ins["no_answer"] == False]
        else:  # multi-answer
            self.data = [
                {"context": ins["plot"].strip(), "question": ins["question"].strip(), "label": [x.strip() for x in ins["answers"]]} for ins in data if ins["no_answer"] == False
            ]

        # debug
        print(f"\n==============================")
        print(f"mode: {mode}, size: {len(self.data)}")
        print(f"==============================\n")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
