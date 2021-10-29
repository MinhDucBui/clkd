from typing import Optional, List, Union
import opustools
from datasets.arrow_dataset import concatenate_datasets
from datasets.load import load_dataset
from pathlib import Path
from src.datamodules.base import BaseDataModule
from torch.utils.data.dataloader import DataLoader
from src.utils.collator import SentenceCollator
from src.utils.utils import get_corresponding_language_pairs


def add_language_tag(x, language_mapping):
    language_tag = [language_mapping["lang_id"][x["language"]][0]]
    return dict(x, **{"language": language_tag})


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
        # TODO: Should be set automatically (coming back after restructuring)
        self.languages = languages

        self.files = {}
        self.data_val = []
        self.val_collate_fn = SentenceCollator(self.tokenizer, self.language_mapping, truncation=True, padding=True)
        # TODO: Change after student cfg
        self.validation_language_mapping = {}

        self.max_length = max_length

    def prepare_data(self):
        # TODO: What if other language shorts?
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

        for index, (language_pair, language_pair_dataset) in enumerate(zip(self.files.keys(), datasets)):
            # TODO: Change after student cfg?
            self.validation_language_mapping[index] = language_pair

            language_pair_dataset = language_pair_dataset.add_column(
                "label", range(len(language_pair_dataset))
            ).rename_column("text", "text_old")

            src = (
                language_pair_dataset.map(lambda x: {"text": x["text_old"].split("\t")[0].strip() if len(x["text_old"].split("\t")) > 1 else None})
                    .remove_columns(["text_old"])
            )
            src = src.filter(lambda example: example['text'] is not None)
            src = src.add_column("language", [self.language_mapping["lang_id"][language_pair.split("_")[0]]] * len(src))

            trg = (
                language_pair_dataset.map(lambda x: {"text": x["text_old"].split("\t")[1].strip() if len(x["text_old"].split("\t")) > 1 else None})
                    .remove_columns(["text_old"])
            )
            trg = trg.filter(lambda example: example['text'] is not None)
            trg = trg.add_column("language", [self.language_mapping["lang_id"][language_pair.split("_")[1]]] * len(trg))

            if stage in (None, "val"):
                self.data_val.append(concatenate_datasets([src, trg]))

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        dataloader_args = {"batch_size": self.batch_size,
                           "num_workers": self.num_workers,
                           "pin_memory": self.pin_memory,
                           "collate_fn": self.val_collate_fn
                           if self.val_collate_fn is not None
                           else self.collate_fn,
                           "shuffle": False}

        return [DataLoader(dataset=data, **dataloader_args) for data in self.data_val]
