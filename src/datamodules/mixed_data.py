from typing import Optional, List, Union
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from src.datamodules.base import BaseDataModule
import hydra
from omegaconf import DictConfig


class MixedDataModule(BaseDataModule):
    def __init__(
            self,
            data_cfg: DictConfig,
            eval_cfg: Optional[DictConfig],
            s_tokenizer: list,
            t_tokenizer,
            train_languages,
            val_languages,
            language_mapping,
            *args,
            **kwargs,
    ):
        # TODO: Change to corresponding Tokenizer. For now, use teacher tokenizer.
        self.tokenizer = t_tokenizer
        self.data_dir = None
        self.collate_fn = None

        # see BaseDataModule
        super().__init__(tokenizer=self.tokenizer, data_dir="None", collate_fn=None, *args, **kwargs)
        # TODO: Should be set automatically (coming back after restructuring)
        self.pairs_of_languages = [["ss", "ht"], ["ss", "mn"]]
        # TODO: Move to cfg
        self.data_dir = Path("C://Users//mducb//Desktop//Master Thesis//project//clkd/data//JW300")

        self.train_datamodule = hydra.utils.instantiate(data_cfg.train,
                                                        s_tokenizer=s_tokenizer,
                                                        t_tokenizer=t_tokenizer,
                                                        languages=train_languages,
                                                        language_mapping=language_mapping,
                                                        )
        self.val_datamodule = hydra.utils.instantiate(data_cfg.val,
                                                      eval_cfg=eval_cfg,
                                                      s_tokenizer=s_tokenizer,
                                                      t_tokenizer=t_tokenizer,
                                                      languages=val_languages,
                                                      language_mapping=language_mapping,
                                                      )
        # TODO: Too much hardcoded. Generalize.
        self.validation_dataset_mapping = self.val_datamodule.validation_dataset_mapping

    def prepare_data(self):
        self.train_datamodule.prepare_data()
        self.val_datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.train_datamodule.setup()
        self.val_datamodule.setup()

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.val_datamodule.val_dataloader()
