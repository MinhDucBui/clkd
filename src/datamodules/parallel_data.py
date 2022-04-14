from typing import Optional
from datasets.load import load_dataset
from src.datamodules.base import BaseDataModule
from datasets.arrow_dataset import Dataset
import os.path

LANG_MAPPING = {"en": "eng", "tr": "tur"}


class ParallelDataModule(BaseDataModule):
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

    def setup(self, stage: Optional[str] = None):
        language_pairs = []
        dataset_lst = {}
        for language in self.languages:
            if language != "en":
                language_pairs.append(["en", "tr"])

        for language_pair in language_pairs:
            joined_language_pair = "-".join(language_pair)
            dir_pair = os.path.join(self.data_dir, joined_language_pair)
            dataset_dict = {"source": [], "target": []}
            for file in os.listdir(dir_pair):
                if len(file.split(".")) == 3:
                    data_language_pair = file.split(".")[1]
                    data_type = file.split(".")[2]
                    if joined_language_pair == data_language_pair:
                        dataset = load_dataset('text', data_files={'train': os.path.join(dir_pair, file)}, split='train')
                        if data_type == "ids":
                            continue
                        elif data_type == "en":
                            dataset_dict["source"] = dataset_dict["source"] + dataset["text"]
                        else:
                            dataset_dict["target"] = dataset_dict["target"] + dataset["text"]

            new_dataset = Dataset.from_dict(
                {
                    'source': dataset_dict["source"],
                    'target': dataset_dict["target"],
                    "l_source": len(dataset_dict["source"]) * [self.language_mapping["lang_id"]["en"]],
                    "l_target": len(dataset_dict["target"]) * [self.language_mapping["lang_id"][language_pair[1]]]
                }
            )
            new_dataset = new_dataset.map(
                lambda x: tokenize_sample(x, self.tokenizer), batched=True).remove_columns(["source", "target",
                                                                                            "l_source", "l_target"])
            dataset_lst[joined_language_pair] = new_dataset

        if stage in (None, "fit"):
            self.data_train = dataset_lst


def tokenize_sample(x, tokenizer):
    s_lang = x["l_source"]
    t_lang = x["l_target"]
    s_x = dict(tokenizer(x["source"], truncation=True, padding=True), **{"language": s_lang})
    s_x = {"s_"+k: v for k, v in s_x.items()}
    t_x = dict(tokenizer(x["target"], truncation=True, padding=True), **{"language": t_lang})
    t_x = {"t_" + k: v for k, v in t_x.items()}
    return dict(**s_x, **t_x)
