from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

@dataclass
class ParallelDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        s_language_batch = []
        t_language_batch = []
        for feature in features:
            s_batch_dict = {}
            t_batch_dict = {}
            for key, value in feature.items():
                if key.split("_")[0] == "s":
                    s_batch_dict["_".join(key.split("_")[1:])] = value
                elif key.split("_")[0] == "t":
                    t_batch_dict["_".join(key.split("_")[1:])] = value
            s_language_batch.append(s_batch_dict)
            t_language_batch.append(t_batch_dict)

        s_batch = self.tokenizer.pad(
            s_language_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        t_batch = self.tokenizer.pad(
            t_language_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in s_batch:
            s_batch["labels"] = s_batch["label"]
            del batch["label"]
        if "label_ids" in s_batch:
            s_batch["labels"] = s_batch["label_ids"]
            del batch["label_ids"]
        if "label" in t_batch:
            t_batch["labels"] = t_batch["label"]
            del batch["label"]
        if "label_ids" in t_batch:
            t_batch["labels"] = t_batch["label_ids"]
            del batch["label_ids"]
        s_batch = {"s_" + k: v for k, v in s_batch.items()}
        t_batch = {"t_" + k: v for k, v in t_batch.items()}

        return {**s_batch, **t_batch}
    