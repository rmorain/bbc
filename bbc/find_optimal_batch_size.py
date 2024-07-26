import torch
from transformers import AutoModel


def find_optimal_batch_size(model, input_len, max_batch_size=1024, start_batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    low, high = start_batch_size, max_batch_size
    optimal_batch_size = start_batch_size

    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()  # Clear GPU cache
            x = torch.ones(mid, input_len, dtype=torch.long).to(device)
            model(x)  # Try a forward pass
            optimal_batch_size = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
            else:
                raise e

    return optimal_batch_size


model = AutoModel.from_pretrained("gpt2")
input_shape = 31
optimal_batch_size = find_optimal_batch_size(model, input_shape)

print("Optimal batch size: ", optimal_batch_size)
