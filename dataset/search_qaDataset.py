from torch.utils.data import Dataset
from datasets import load_dataset


class search_qaDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset("search_qa", "train_test_val")

        if self.mode == "train":
            self.data = self.data["train"]
        elif self.mode == "valid":
            self.data = self.data["validation"]
        else:  # test
            self.data = self.data["test"]

        data = [row for row in self.data]

        if self.mode == "train":
            self.data = [
                {
                    "context": " ".join(list(filter(None, list(dict.fromkeys(ins["search_results"]["snippets"]))))),
                    "question": ins["question"].strip(),
                    "label": ins["answer"].strip(),
                }
                for ins in data
            ]
        else:  # multi-answer
            self.data = [
                {
                    "context": " ".join(list(filter(None, list(dict.fromkeys(ins["search_results"]["snippets"]))))),
                    "question": ins["question"].strip(),
                    "label": [ins["answer"].strip()],
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
