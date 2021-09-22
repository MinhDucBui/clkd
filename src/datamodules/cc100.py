from typing import Optional
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorForLanguageModeling
from pathlib import Path
from src.models.modules.get_model_architecture import get_tokenizer
import os.path
import sys


class CC100DataModule(LightningDataModule):
    def __init__(
            self,
            teacher_model_type: str,
            languages: list,
            batch_size: int = 8,
            data_dir: str = "./data/cc100/",
            max_length: int = 100,
            num_workers: int = 8,
            pin_memory: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = str(Path(data_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.pin_memory = pin_memory
        self.tokenizer = get_tokenizer(self.hparams)

        self.languages = languages
        self.paths_to_files = []
        self.construct_path_to_files()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        train_data = self.load_dataset_iterable()

        self.data_train = train_data
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=0.15)

    def construct_path_to_files(self):
        """Check and Construct Paths to languages.

        Returns:

        """

        # We assume that file is a txt file
        file_type = ".txt"

        for single_language in self.languages:

            # Construct path to language
            file_path = os.path.join(self.data_dir, single_language + file_type)

            # Check if file exists
            if not os.path.isfile(file_path):
                sys.exit("Path {} does not exist! Please download and extract missing file.".format(file_path))

            self.paths_to_files.append(file_path)

        # Check if any files were added or not
        if not self.paths_to_files:
            sys.exit("No files were specified. Please check the variable languages and data_dir in cc100 datamodule.")


    @property
    def num_labels(self) -> int:
        return self.tokenizer.vocab_size

    #def __len__(self):
    #    return len(self.data_train) if self.data_train is not None else 0

    # TODO: Move to Collator
    def load_dataset_iterable(self):
        dataset = load_dataset('text', data_files={'train': self.paths_to_files}, split='train', streaming=True)

        # https://github.com/huggingface/datasets/issues/2583
        dataset = dataset.with_format("torch")

        # Shuffle Dataset
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

        # Tokenize Dataset
        tokenized_dataset = dataset.map(lambda x: self.tokenizer(x["text"]))

        return tokenized_dataset

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to
        be done only from a single process in distributed settings, e.g.,
        download, tokenize, etc...

        """

        return None

    def setup(self, stage: Optional[str] = None):
        """There are also data operations you might want to perform on every GPU. Use setup to do things like:
        count number of classes, build vocabulary, perform train/val/test splits,
        apply transforms (defined explicitly in your datamodule), etc...

        """

        pass


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator
        )

    # TODO: Change Val and test dataloader to a unseen dataset
    def val_dataloader(self):
        self.data_train = self.data_train.with_format("torch")
        dataloader = DataLoader(
            dataset=self.data_train,
            sampler=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator
        )

        return dataloader

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator
        )