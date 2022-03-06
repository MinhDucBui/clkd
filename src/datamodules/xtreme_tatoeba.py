from typing import Optional
from datasets.load import load_dataset_builder
from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset
from src.datamodules.base import BaseDataModule
from src.utils.utils import add_language_tag_tokenizer
import hydra
from datasets.arrow_dataset import Dataset

LANG_MAPPING = {"en": "eng", "tr": "tur"}


class XtremeTatoebaDataModule(BaseDataModule):
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

    def setup(self, stage: Optional[str] = None):
        split_samples = '{}[0:{}]'.format("validation", self.max_length)
        if "en" == self.languages[0]:
            source_lang = LANG_MAPPING[self.languages[1]]
            self.languages[1], self.languages[0] = self.languages[0], self.languages[1]
        else:
            source_lang = LANG_MAPPING[self.languages[0]]

        dataset = load_dataset("xtreme", "tatoeba." + source_lang, split=split_samples)
        print(dataset)
        print(self.languages)
        dataset = preprocess_xtreme_tatoeba(dataset, self.tokenizer, self.languages[0], self.languages[1],
                                            self.language_mapping, source=True)
        print(dataset)

        if stage in (None, "val"):
            self.data_val = dataset


def preprocess_xtreme_tatoeba(dataset, tokenizer, src_lang, trg_lang, language_mapping, start_index=0, source=True):
    new_dataset = Dataset.from_dict(
        {
            'text': [i.replace("\n", "") for i in dataset["source_sentence"]]
                    + [i.replace("\n", "") for i in dataset["target_sentence"]],
            "language": len(list(dataset["source_sentence"])) * [language_mapping["lang_id"][src_lang]]
                        + len(list(dataset["target_sentence"])) * [language_mapping["lang_id"][trg_lang]]
        }
    )

    new_dataset = new_dataset.add_column("labels", [x for x in range(len(new_dataset))])
    new_dataset = new_dataset.map(
        lambda x: tokenize_sample(x, tokenizer)).remove_columns(["text"])
    return new_dataset


def tokenize_sample(x, tokenizer):
    language_tag = x["language"]
    x = dict(tokenizer(x["text"], truncation=True, padding=True))
    return dict(x, **{"language": language_tag})
