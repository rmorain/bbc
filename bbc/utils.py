import torch
from trl.core import LengthSampler

from datasets import Dataset, load_from_disk


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def contiguous_copy(str_list, n):
    contiguous_list = []
    for x in str_list:
        row = [x] * n
        contiguous_list += row
    return contiguous_list


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


def find_optimal_batch_size(
    model, input_shape, max_batch_size=1024, start_batch_size=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    low, high = start_batch_size, max_batch_size
    optimal_batch_size = start_batch_size

    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()  # Clear GPU cache
            x = torch.ones(mid, *input_shape).to(device)
            model(x)  # Try a forward pass
            optimal_batch_size = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
            else:
                raise e

    return optimal_batch_size
