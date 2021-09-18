import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.utils.collator import SentencePairCollator


class PlainDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, list]):
        if isinstance(idx, list):
            return [self.data[i] for i in idx]
        return self.data[idx]


class MNLIDataModule(LightningDataModule):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 8,
        data_dir: str = "./data/multinli_1.0/",
        max_length: int = 100,
        num_workers: int = 8,
        pin_memory: bool = True,
        # possibly downsampling dataset in stratified manner
        sample_seed: int = 42,
        sample_with_replace: bool = False,
        samples_per_cls: int = 0,
        # functionality for contrastive learning
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.samples_per_cls = samples_per_cls
        self.sample_with_replace = sample_with_replace
        self.sample_seed = sample_seed

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_labels(self) -> int:
        return 3

    def __len__(self):
        return len(self.data_train) if self.data_train is not None else 0

    @staticmethod
    def read_mnli(
        path: Union[str, Path] = "./datasets/multinli_1.0/multinli_1.0_train.jsonl",
        return_dataframe: bool = False,
        samples_per_cls: int = 0,
        sample_with_replace: bool = False,
        sample_seed: int = 42,
        *args,
        **kwargs,
    ):
        df = pd.read_json(path, lines=True)
        valid_labels = ["contradiction", "entailment", "neutral"]
        if df["gold_label"].dtype == "object":
            replace_dico = {label: i for i, label in enumerate(valid_labels)}
            # filter non-existing gold labels
            df = df[df["gold_label"].isin(valid_labels)]
            df["gold_label"] = df["gold_label"].replace(replace_dico)
        # some instances have trailing whitespaces
        df["sentence1"] = df["sentence1"].str.strip()
        df["sentence2"] = df["sentence2"].str.strip()

        if samples_per_cls:
            df = df.groupby("gold_label").apply(
                lambda x: x.sample(
                    min(len(x), samples_per_cls),
                    replace=sample_with_replace,
                    random_state=sample_seed,
                )
            )
        df = df.loc[:, ["sentence1", "sentence2", "gold_label"]]
        data = df.values.tolist()
        return data

    def setup(self, stage: Optional[str] = None):
        train_data = self.read_mnli(
            path=os.path.join(self.data_dir, "multinli_1.0_train.jsonl"),
            return_dataframe=False,
            samples_per_cls=self.samples_per_cls,
            sample_with_replace=self.sample_with_replace,
            sample_seed=self.sample_seed,
        )
        self.data_train = PlainDataset(train_data)

        self.collator = SentencePairCollator(
            model_name_or_path=self.model_name_or_path,
            max_length=self.max_length,
        )
        test_data = self.read_mnli(
            path=os.path.join(self.data_dir, "multinli_1.0_dev_mismatched.jsonl"),
            return_dataframe=False,
            samples_per_cls=0,
        )
        self.data_test = PlainDataset(test_data)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )
