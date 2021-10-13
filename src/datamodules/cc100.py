from typing import Optional
from datasets import load_dataset, interleave_datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, ChainDataset
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorForLanguageModeling
import copy
from pathlib import Path
from itertools import islice
import os.path
import sys
import requests
from src.utils import utils
import lzma, shutil
import hydra

log = utils.get_logger(__name__)

SEED = 42


def add_language_tag(dataset, language):
    return dataset.map(lambda x: dict(x, **{"language": language}))


def add_language_tag_tokenizer(x, tokenizer, language_mapping):
    language_tag = [language_mapping["lang_id"][x["language"]][0]]
    x = dict(tokenizer(x["text"]))
    return dict(x, **{"language": language_tag})


class CC100DataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            language_mapping: dict,
            s_tokenizer: list,
            t_tokenizer,
            **kwargs,
    ):
        super().__init__()
        #self.save_hyperparameters()
        self.data_dir = str(Path(data_dir))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.language_mapping = language_mapping
        self.languages = list(language_mapping["lang_id"].keys())
        self.s_tokenizer = s_tokenizer
        self.t_tokenizer = t_tokenizer
        # TODO: Different Tokenizer for student/teacher
        self.tokenizer = self.t_tokenizer

        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=0.15)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_labels(self) -> int:
        return self.tokenizer.vocab_size

    # def __len__(self):
    #    return len(self.data_train) if self.data_train is not None else 0

    # TODO: Move to Collator
    def load_dataset_iterable(self, paths_to_files):

        dataset_lst = []
        for language, path in paths_to_files.items():
            language_dataset = load_dataset('text', data_files={'train': path}, split='train', streaming=True)
            language_dataset = add_language_tag(language_dataset, language)
            # Shuffle Dataset
            dataset_lst.append(language_dataset)

        dataset = interleave_datasets(dataset_lst)  # , probabilities=[0.8, 0.2], seed=42)

        # https://github.com/huggingface/datasets/issues/2583
        dataset = dataset.with_format("torch")

        # TODO: This Part should happen in Collator
        tokenized_dataset = dataset.map(
            lambda x: add_language_tag_tokenizer(x, self.tokenizer, self.language_mapping))

        return tokenized_dataset

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to
        be done only from a single process in distributed settings, e.g.,
        download, tokenize, etc...

        """

        # We assume that file is a txt or txt.xz file
        file_type = ".txt"
        file_type_compressed = ".txt.xz"

        # Save the paths to the data
        paths_to_files = {}

        # Loop through the languages
        for single_language in self.languages:
            if single_language == "False":
                continue

            # Construct path to language
            file_path_txt = os.path.join(self.data_dir, single_language + file_type)
            file_path_compressed = os.path.join(self.data_dir, single_language + file_type_compressed)

            # Check if file exists
            if not os.path.isfile(file_path_txt):
                if not os.path.isfile(file_path_compressed):
                    log.info("No txt or txt.xz file for {} exist! Proceed to download file...".format(single_language))
                    self.download_file(single_language, self.data_dir)
                self.decompress_xz(file_path_compressed)
            paths_to_files[single_language] = file_path_txt

        self.data_train = self.load_dataset_iterable(paths_to_files)

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

    def download_file(self, language, output_folder):

        file_name = '{language}.txt.xz'.format(language=language)
        link = 'http://data.statmt.org/cc-100/' + file_name
        output_file = os.path.join(output_folder, file_name)
        print("\n")
        with open(output_file, "wb") as f:
            log.info("Downloading %s" % file_name)
            response = requests.get(link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                log.info("No File Length found. Can not display progress bar.")
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                chunk_size = int(total_length / 100)
                for data in response.iter_content(chunk_size):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50 - done), done * 2))
                    sys.stdout.flush()
        print("\n")

    def decompress_xz(self, input_file):
        log.info("Decompress %s" % input_file)
        input_file = Path(input_file)
        destination_dir = os.path.dirname(input_file)
        with lzma.open(input_file) as compressed:
            output_path = Path(destination_dir) / input_file.stem
            with open(output_path, 'wb') as destination:
                try:
                    shutil.copyfileobj(compressed, destination)
                except EOFError:
                    log.info("File {} is corrupted. Please Delete the file.".format(input_file))
