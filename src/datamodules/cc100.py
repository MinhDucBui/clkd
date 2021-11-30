from typing import Optional
from datasets import load_dataset, interleave_datasets
import os.path
from src.utils.utils import get_logger, download_file, decompress_xz, add_language_tag_tokenizer, add_language_tag
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

        # Path to the downloaded data for each language
        self.paths_to_files = {}

    @property
    def num_labels(self) -> int:
        return self.tokenizer.vocab_size

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.data_train = self.load_dataset_iterable(self.paths_to_files)

    def prepare_data(self):
        """Use this method to do things that might write to disk or that need to
        be done only from a single process in distributed settings, e.g.,
        download, tokenize, etc...

        """

        # Get Training Data from cc100 for MLM
        log.info("Prepare/Download Training Data...")

        # We assume that file is a txt or txt.xz file
        file_type = ".txt"
        file_type_compressed = ".txt.xz"

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
                    log.info("Downloading %s" % single_language)
                    download_file(single_language, self.data_dir)
                log.info("Start Decompressing %s" % file_path_compressed)
                decompress_xz(file_path_compressed)

    def construct_path_to_files(self):
        file_type = ".txt"
        for single_language in self.languages:
            file_path_txt = os.path.join(self.data_dir, single_language + file_type)
            self.paths_to_files[single_language] = file_path_txt

    # TODO: Move to Collator
    def load_dataset_iterable(self, paths_to_files):
        self.construct_path_to_files()
        dataset_lst = {}
        for language, path in paths_to_files.items():
            language_dataset = load_dataset('text', data_files={'train': path}, split='train', streaming=True)
            language_dataset = add_language_tag(language_dataset, language)

            # https://github.com/huggingface/datasets/issues/2583
            language_dataset = language_dataset.with_format("torch")

            # TODO: This Part should happen in Collator?
            tokenized_dataset = language_dataset.map(
                lambda x: add_language_tag_tokenizer(x, self.tokenizer, self.language_mapping))

            dataset_lst[language] = tokenized_dataset
        return dataset_lst

