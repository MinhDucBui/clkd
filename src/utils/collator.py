import torch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from src.utils import flatten_dict


def add_language_tag_tokenizer(x, tokenizer, language_mapping):
    language_tag = [language_mapping["lang_id"][x["language"]][0]]
    x = dict(tokenizer(x["text"], truncation=True, padding=True))
    return dict(x, **{"language": language_tag})


class SentenceCollator:
    def __init__(
            self, tokenizer, **kwargs,
    ):
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.kwargs: dict = kwargs

    def __call__(self, inputs) -> BatchEncoding:
        merged_inputs = flatten_dict(inputs)
        batch: BatchEncoding = self.tokenizer(merged_inputs["text"], truncation=True, padding=True, return_tensors="pt")
        if "labels" in merged_inputs:
            batch["labels"] = torch.tensor(merged_inputs["labels"], dtype=torch.long)
        if "language" in merged_inputs:
            batch["language"] = torch.tensor(merged_inputs["language"], dtype=torch.long)
        return batch
