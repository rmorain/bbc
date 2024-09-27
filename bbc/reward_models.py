from typing import Dict, List, Optional, Union

import torch
from transformers import pipeline


class RewardModel:
    """
    A parent class for models that generate reward lists from strings.

    This class provides a basic framework for reward models that can handle
    either a single string or a batch of strings as input and output a list
    of rewards (floats) for each input string. Subclasses can implement
    specific reward calculation logic.
    """

    def __init__(self):
        """
        Initialize the reward model.

        (Optional) Subclasses can override this method to perform any necessary
        initialization.
        """
        pass

    def __call__(
        self, input_string: Union[str, List[str]]
    ) -> List[Optional[List[float]]]:
        """
        Calculate and return a list of rewards based on the input string(s).

        Args:
          input_string (Union[str, List[str]]): A single string or a list of strings
            representing the input for reward calculation.

        Returns:
          List[Optional[List[float]]]: A list containing reward lists (floats)
            for each input string. If the calculation fails for a particular
            input string, the corresponding element in the list will be None.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method")

    def to(self, device: torch.device) -> None:
        """
        Set the device for the RewardModel.

        Args:
            device (torch.device): The device to use for the RewardModel.

        Returns:
            None
        """

        raise NotImplementedError("Subclasses must implement the to method")


class SentimentRewardModel(RewardModel):
    """
    A subclass of `RewardModel` that calculates a reward using the default
    Hugging Face sentiment classifier.
    """

    def __init__(
        self,
        model: Optional[
            str
        ] = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[Union[torch.device, int]] = -1,
        kwargs: Optional[Dict] = {
            "top_k": 2,
            "function_to_apply": None,
            "batch_size": 16,
        },
    ):
        """
        Initialize the sentiment reward model.

        Args:
          model (Optional[str] =
          "distilbert/distilbert-base-uncased-finetuned-sst-2-english"):
            The Hugging Face model identifier for sentiment analysis.
          device (Optional[Union[torch.device, int]] = -1):
            The device to use for the sentiment analysis pipeline. This can be
            either a `torch.device` object or an integer representing the GPU
            index (if using CUDA with PyTorch).
          kwargs (Optional[Dict]): Additional keyword arguments to be passed
            to the Hugging Face `pipeline` function.
        """
        self.reward_model = pipeline(
            "sentiment-analysis",
            model=model,
            device=device,
        )
        self.kwargs = kwargs
        self.device = device

    def __call__(
        self, input_string: Union[str, List[str]]
    ) -> List[Optional[List[float]]]:
        """
        Assigns scores for each input string.

        Args:
            input_string (Union[str, List[str]]): A List (batch size) of strings to be
                evaluated.

        Returns:
            List[Optional[List[float]]]: A List (batch size) of Lists (number of
                classes) containing scores.
        """
        prediction = self.reward_model(input_string, **self.kwargs)
        scores = self.process_sentiment_output(prediction)
        return scores

    def to(self, device: torch.device) -> None:
        """
        Set the device for the RewardModel.

        Args:
            device (torch.device): The device to use for the RewardModel.

        Returns:
            None
        """
        self.reward_model.model = self.reward_model.model.to(device)
        self.reward_model.device = device
        self.device = device
        return self

    def process_sentiment_output(self, model_output):
        return [
            [
                next(item["score"] for item in sample if item["label"] == "NEGATIVE"),
                next(item["score"] for item in sample if item["label"] == "POSITIVE"),
            ]
            for sample in model_output
        ]
