from typing import Optional
from datasets.load import load_dataset_builder
from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset
from src.datamodules.base import BaseDataModule
from src.utils.utils import add_language_tag_tokenizer
import hydra

LANGUAGE_MAPPING = {"sw": "swh", "en": "en", "tr": "tr"}


class TatoebaDataModule(BaseDataModule):
    def __init__(
            self,
            s_tokenizer: list,
            t_tokenizer,
            languages,
            language_mapping,
            max_length: int,
            *args,
            **kwargs,
    ):
        # TODO: Change to corresponding Tokenizer. For now, use teacher tokenizer.
        self.tokenizer = t_tokenizer
        self.language_mapping = language_mapping
        # see BaseDataModule
        super().__init__(tokenizer=self.tokenizer, *args, **kwargs)
        self.languages = languages

        self.files = {}
        self.data_val = []

        self.validation_dataset_mapping = {}
        self.max_length = max_length

    def prepare_data(self):
        """
        Download MNLI from Huggingface datasets hub.
        See: https://huggingface.co/datasets/glue
        """
        # download with Huggingface datasets
        try:
            dataset = load_dataset_builder("tatoeba", 
                                           lang1=LANGUAGE_MAPPING[self.languages[0]], 
                                           lang2=LANGUAGE_MAPPING[self.languages[1]])
            dataset.download_and_prepare()
        except FileNotFoundError:
            self.languages[1], self.languages[0] = self.languages[0], self.languages[1]
            dataset = load_dataset_builder("tatoeba", 
                                           lang1=LANGUAGE_MAPPING[self.languages[0]], 
                                           lang2=LANGUAGE_MAPPING[self.languages[1]])
            dataset.download_and_prepare()

    def setup(self, stage: Optional[str] = None):

        split_samples = '{}[0:{}]'.format("train", self.max_length)
        try:
            dataset = load_dataset("tatoeba", 
                                   lang1=LANGUAGE_MAPPING[self.languages[0]], 
                                   lang2=LANGUAGE_MAPPING[self.languages[1]], split=split_samples)
        except FileNotFoundError:
            self.languages[1], self.languages[0] = self.languages[0], self.languages[1]
            dataset = load_dataset("tatoeba", 
                                   lang1=LANGUAGE_MAPPING[self.languages[0]], 
                                   lang2=LANGUAGE_MAPPING[self.languages[1]], split=split_samples)

        src = preprocess_tatoeba(dataset, self.tokenizer, self.languages[0], self.language_mapping)
        trg = preprocess_tatoeba(dataset, self.tokenizer, self.languages[1],
                                 self.language_mapping,
                                 start_index=len(src))

        if stage in (None, "val"):
            self.data_val = concatenate_datasets([src, trg])


def preprocess_tatoeba(dataset, tokenizer, language, language_mapping, start_index=0):
    def split_text(x, language: str):
        return {"text": x["translation"][language]}
    
    new_dataset = dataset.map(lambda x: split_text(x, LANGUAGE_MAPPING[language]))
    new_dataset = new_dataset.remove_columns(["translation"]).remove_columns(["id"])
    new_dataset = new_dataset.filter(lambda example: example['text'] is not None)
    new_dataset = new_dataset.add_column("language", [language] * len(new_dataset))
    new_dataset = new_dataset.add_column("labels", [x for x in range(start_index, len(new_dataset)+start_index)])
    new_dataset = new_dataset.map(
        lambda x: add_language_tag_tokenizer(x, tokenizer, language_mapping)).remove_columns(["text"])
    return new_dataset