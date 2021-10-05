from transformers import BertConfig, BertForMaskedLM
from omegaconf import DictConfig
from pytorch_lightning.utilities.parsing import AttributeDict
from typing import Any
from src.models.base_model import BaseModel
from typing import Union


class BertForDistill(BaseModel):
    hparams: AttributeDict

    def __init__(
        self,
        pretrained_model_name_or_path: Union[bool, str],
        cfg: DictConfig,
        **kwargs: Any,
    ):
        self.save_hyperparameters()
        super().__init__(pretrained_model_name_or_path)

        if not self.model:
            architecture_config = BertConfig(**cfg)
            self.model = BertForMaskedLM(architecture_config)

        # ToDo: For now, just inherit forward method
        forward = self.model.forward

