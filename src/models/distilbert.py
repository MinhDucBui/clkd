from transformers import DistilBertConfig, DistilBertForMaskedLM
from omegaconf import DictConfig
from pytorch_lightning.utilities.parsing import AttributeDict
from typing import Any
from src.models.base_model import BaseModel
from typing import Union


class DistillBertForDistill(DistilBertForMaskedLM, BaseModel):
    hparams: AttributeDict

    def __init__(
        self,
        pretrained_model_name_or_path: Union[bool, str],
        cfg: DictConfig,
        **kwargs: Any,
    ):
        self.save_hyperparameters()
        super(DistilBertForMaskedLM, self).__init__(BaseModel)

        if not self.model:
            architecture_config = DistilBertConfig(**cfg)
            self.model = DistilBertForMaskedLM(architecture_config)

        # ToDo: For now, just inherit forward method
        forward = self.model.forward