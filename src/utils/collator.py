from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


class SentencePairCollator:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 128,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )

    def __call__(self, *args, **kwds):
        batch = self.tokenize(*args, **kwds)
        return batch

    def tokenize(self, inputs: List[Tuple[str, str, int]]):
        sentence1, sentence2, labels = zip(*inputs)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        batch = self.tokenizer(
            text=list(sentence1),
            text_pair=list(sentence2),
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.from_numpy(labels)
        return batch
