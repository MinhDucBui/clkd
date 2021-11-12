from typing import Optional
import opustools
from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset
from pathlib import Path
from src.datamodules.base import BaseDataModule
from src.utils.utils import add_language_tag_tokenizer
import sys
import hydra



class JW300DataModule(BaseDataModule):
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
        # TODO: OPUS is down...
        for pair_language in self.languages:
            path_folder = Path(self.data_dir) / Path("_".join(pair_language))
            Path.mkdir(path_folder, parents=True, exist_ok=True)
            file_name = self.data_dir.__str__() + "/" + pair_language[0] + "_" + pair_language[1] + ".txt"
            if Path(file_name).is_file():
                self.files["_".join(pair_language)] = file_name
                continue
            # download with opustools
            opus_reader = opustools.OpusRead(
                directory="JW300",
                source="ss",
                target="mn",
                write_mode="moses",
                write=[file_name],
                download_dir=path_folder.__str__(),
                suppress_prompts=True
            )
            opus_reader.printPairs()
            self.files["_".join(pair_language)] = file_name

    def setup(self, stage: Optional[str] = None):
        split_samples = []
        for key in self.files.keys():
            split_samples.append('{}[0:{}]'.format(key, self.max_length))
        datasets = load_dataset('text',
                                data_files={key: file for key, file in self.files.items()},
                                split=split_samples)

        index = 0
        # Is being applied to ALL datasets, but should only be applied to corresponding task dataset
        # TODO: Change this behaviour
        for language_pair, language_pair_dataset in zip(self.files.keys(), datasets):
            language_pair_dataset = language_pair_dataset.rename_column("text", "text_old")

            src = preprocess_jw300(language_pair_dataset, self.tokenizer, language_pair, self.language_mapping, "src")
            trg = preprocess_jw300(language_pair_dataset, self.tokenizer, language_pair, self.language_mapping, "trg")

            if stage in (None, "val"):
                for task_name in self.eval_cfg.keys():
                    # TODO: Change after student cfg?
                    self.validation_dataset_mapping[index] = {"languages": language_pair,
                                                              "task": task_name}
                    index += 1

                    self.val_collate_fn.append(hydra.utils.instantiate(self.val_collate_fn_dict[task_name],
                                                                       tokenizer=self.tokenizer)())
                    self.data_val.append(concatenate_datasets([src, trg]))


def preprocess_jw300(dataset, tokenizer, language_pair, language_mapping, direction: str):
    def split_text(x, direction: str):
        if direction == "src":
            return {"text": x["text_old"].split("\t")[0].strip() if len(x["text_old"].split("\t")) > 1 else None}
        elif direction == "trg":
            return {"text": x["text_old"].split("\t")[1].strip() if len(x["text_old"].split("\t")) > 1 else None}
        else:
            sys.exit("Direction has to be src or trg.")

    new_dataset = (dataset.map(lambda x: split_text(x, direction)).remove_columns(["text_old"]))
    new_dataset = new_dataset.filter(lambda example: example['text'] is not None)

    if direction == "src":
        new_dataset = new_dataset.add_column("language", [language_pair.split("_")[0]] * len(new_dataset))
    elif direction == "trg":
        new_dataset = new_dataset.add_column("language", [language_pair.split("_")[1]] * len(new_dataset))

    new_dataset = new_dataset.map(
        lambda x: add_language_tag_tokenizer(x, tokenizer, language_mapping)).remove_columns(["text"])
    return new_dataset
