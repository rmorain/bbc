from transformers import AutoTokenizer

from datasets import Dataset


def create_tokenized_dataset(
    target: str,
    target_label: str,
    prompt: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 128,
    repetitions: int = 1,
) -> Dataset:
    """
    Create a Hugging Face Dataset with tokenized fields.

    Args:
        query (str): The query string.
        target (str): The target string.
        target_label (str): The label for the target.
        prompt (str): The prompt string.
        tokenizer_name (str): Name of the tokenizer to use.
        max_length (int): Maximum token length for truncation/padding.
        repetitions (int): Number of times to repeat the example.

    Returns:
        Dataset: A Hugging Face Dataset object containing the tokenized example(s).
    """
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Define a helper function to tokenize a single text input
    def tokenize(text):
        return tokenizer.encode(
            text,
        )

    # Tokenize each field
    tokenized_query = [
        tokenize(f"Sentiment {target_label}. {prompt}") for _ in range(repetitions)
    ]
    repeated_target = [target for _ in range(repetitions)]
    tokenized_target_label = [target_label for _ in range(repetitions)]
    tokenized_prompt = [tokenize(prompt) for _ in range(repetitions)]

    # Create the dataset dictionary with tokenized fields
    data = {
        "query": tokenized_query,
        "target": repeated_target,
        "target_label": tokenized_target_label,
        "prompt": tokenized_prompt,
    }

    return Dataset.from_dict(data)


# Example usage:
if __name__ == "__main__":
    target_example = 0
    target_label_example = "negative"
    prompt_example = "Hello, world!"

    dataset = create_tokenized_dataset(
        target=target_example,
        target_label=target_label_example,
        prompt=prompt_example,
        tokenizer_name="gpt2",
        max_length=32,
        repetitions=256 * 8,
    )

    dataset.set_format(type="torch")
    dataset.save_to_disk("datasets/hello_world_2048")
