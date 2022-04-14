from typing import Optional, List, Union
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule
import hydra


class MixedDataModule(LightningDataModule):
    def __init__(
            self,
            cfg_datamodule,
            language_mapping,
            student_tokenizers,
            teacher_tokenizer,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.train_datamodule = hydra.utils.instantiate(cfg_datamodule.train,
                                                        languages=list(language_mapping["id_lang"].values()),
                                                        language_mapping=language_mapping,
                                                        s_tokenizer=student_tokenizers,
                                                        t_tokenizer=teacher_tokenizer)

        if "train_parallel_data" in cfg_datamodule.keys():
            self.parallel_data = True
            self.train_parallel_datamodule = hydra.utils.instantiate(cfg_datamodule.train_parallel_data,
                                                                     languages=list(
                                                                         language_mapping["id_lang"].values()),
                                                                     language_mapping=language_mapping,
                                                                     s_tokenizer=student_tokenizers,
                                                                     t_tokenizer=teacher_tokenizer)
        else:
            self.parallel_data = False

        # Init Val Dataset
        self.val_datamodules = []
        for key, value in cfg_datamodule.items():
            if "val" in key:
                val_datamodule = hydra.utils.instantiate(value,
                                                         languages=list(language_mapping["id_lang"].values()),
                                                         language_mapping=language_mapping,
                                                         s_tokenizer=student_tokenizers,
                                                         t_tokenizer=teacher_tokenizer)
                self.val_datamodules.append(val_datamodule)

    def prepare_data(self):
        if self.parallel_data:
            self.train_parallel_datamodule.prepare_data()
        self.train_datamodule.prepare_data()
        for val_datamodule in self.val_datamodules:
            val_datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        if self.parallel_data:
            self.train_parallel_datamodule.setup()
        self.train_datamodule.setup()
        for val_datamodule in self.val_datamodules:
            val_datamodule.setup()

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], dict[DataLoader]]:
        train_data = self.train_datamodule.train_dataloader()
        if self.parallel_data:
            train_parallel = self.train_parallel_datamodule.train_dataloader()
            return {**train_parallel, **train_data}
        return self.train_datamodule.train_dataloader()

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_dataloaders = [loader.val_dataloader() for loader in self.val_datamodules]
        return val_dataloaders
