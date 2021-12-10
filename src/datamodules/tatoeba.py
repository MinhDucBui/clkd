from typing import Optional
from datasets.load import load_dataset_builder
from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset
from src.datamodules.base import BaseDataModule
from src.utils.utils import add_language_tag_tokenizer
import hydra


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
        for language_pair in self.languages:
            try:
                dataset = load_dataset_builder("tatoeba", lang1=language_pair[0], lang2=language_pair[1])
                dataset.download_and_prepare()
            except FileNotFoundError:
                language_pair[1], language_pair[0] = language_pair[0], language_pair[1]
                dataset = load_dataset_builder("tatoeba", lang1=language_pair[0], lang2=language_pair[1])
                dataset.download_and_prepare()

    def setup(self, stage: Optional[str] = None):

        split_samples = '{}[0:{}]'.format("train", self.max_length)
        index = 0
        for language_pair in self.languages:
            try:
                dataset = load_dataset("tatoeba", lang1=language_pair[0], lang2=language_pair[1], split=split_samples)
            except FileNotFoundError:
                language_pair[1], language_pair[0] = language_pair[0], language_pair[1]
                dataset = load_dataset("tatoeba", lang1=language_pair[0], lang2=language_pair[1], split=split_samples)
            src = preprocess_tatoeba(dataset, self.tokenizer, language_pair[0], self.language_mapping)
            trg = preprocess_tatoeba(dataset, self.tokenizer, language_pair[1], self.language_mapping)

            if stage in (None, "val"):
                for task_name in self.eval_cfg.keys():
                    # TODO: Change after student cfg?
                    self.validation_dataset_mapping[index] = {"languages": "_".join(language_pair),
                                                              "task": task_name}
                    index += 1

                    self.val_collate_fn.append(hydra.utils.instantiate(self.val_collate_fn_dict[task_name],
                                                                       tokenizer=self.tokenizer)())
                    self.data_val.append(concatenate_datasets([src, trg]))


def preprocess_tatoeba(dataset, tokenizer, language, language_mapping):
    def split_text(x, language: str):
        return {"text": x["translation"][language]}

    new_dataset = dataset.map(lambda x: split_text(x, language))
    new_dataset = new_dataset.remove_columns(["translation"]).remove_columns(["id"])
    new_dataset = new_dataset.filter(lambda example: example['text'] is not None)
    new_dataset = new_dataset.add_column("language", [language] * len(new_dataset))
    new_dataset = new_dataset.map(
        lambda x: add_language_tag_tokenizer(x, tokenizer, language_mapping)).remove_columns(["text"])
    return new_dataset