from trl.core import LengthSampler

from datasets import Dataset, load_from_disk


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


class BBCDataset(Dataset):
    def __init__(
        self,
        ds_path,
        emotions=["negative", "positive"],
        tokenizer=None,
    ):
        self.ds_path = ds_path
        self.emotions = emotions
        self.tokenizer = tokenizer
        try:
            self.ds = load_from_disk(self.ds_path + "_tokenized")
        except FileNotFoundError:
            self.ds = self._build_dataset()
            self.ds.save_to_disk(self.ds_path + "_tokenized")

    def _build_dataset(self, input_min_text_length=2, input_max_text_length=8):
        ds = load_from_disk(self.ds_path)

        self.input_size = LengthSampler(input_min_text_length, input_max_text_length)
        ds = ds.map(self._tokenize, batched=False)
        ds = ds.remove_columns(["text", "label"])

        ds.set_format(type="torch")
        return ds

    def _tokenize(self, sample):
        if sample["label"] == 0:
            sample["target"] = 1
        else:
            sample["target"] = 0
        sample["target_label"] = self.emotions[sample["target"]]
        input_size = self.input_size()
        sample["prompt"] = self.tokenizer.encode(sample["text"])[:input_size]
        sample["query"] = self.tokenizer.encode(
            f"Sentiment: {self.emotions[sample['target']]}. {self.tokenizer.decode(sample['prompt'])}"
        )
        return sample

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return self.ds[index]

    def save_to_disk(self, path):
        self.ds.save_to_disk(path)
