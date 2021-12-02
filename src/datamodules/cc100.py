from typing import Optional
from datasets import load_dataset
import os.path
from src.utils.utils import get_logger, add_language_tag_tokenizer, add_language_tag
from src.datamodules.base import BaseDataModule

log = get_logger(__name__)

SEED = 42


class CC100DataModule(BaseDataModule):
    def __init__(
            self,
            s_tokenizer: list,
            t_tokenizer,
            languages,
            language_mapping,
            *args,
            **kwargs,
    ):

        # TODO: Different Tokenizer for student/teacher
        super().__init__(tokenizer=t_tokenizer, *args, **kwargs)

        self.language_mapping = language_mapping
        self.languages = languages
        self.s_tokenizer = s_tokenizer
        self.t_tokenizer = t_tokenizer
        # TODO: Different Tokenizer for student/teacher
        self.tokenizer = self.t_tokenizer


    @property
    def num_labels(self) -> int:
        return self.tokenizer.vocab_size

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.data_train = self.load_dataset_iterable()

    # TODO: Move to Collator
    def load_dataset_iterable(self):
        """Download with Hugging Face"""

        dataset_lst = {}
        for single_language in self.languages:
            if single_language == "False":
                continue

            log.info("Downloading %s" % single_language)
            language_dataset = load_dataset("cc100", lang=single_language, split='train', streaming=True)
            language_dataset = add_language_tag(language_dataset, single_language)

            # https://github.com/huggingface/datasets/issues/2583
            language_dataset = language_dataset.with_format("torch")

            # TODO: This Part should happen in Collator?
            tokenized_dataset = language_dataset.map(
                lambda x: add_language_tag_tokenizer(x, self.tokenizer, self.language_mapping))

            dataset_lst[single_language] = tokenized_dataset
        return dataset_lst
