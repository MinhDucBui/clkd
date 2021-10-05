from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM
from omegaconf import DictConfig
from pytorch_lightning.utilities.parsing import AttributeDict
from typing import Any
from src.models.base_model import BaseModel
from typing import Union


class XLMRForDistill(BaseModel):
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
            architecture_config = XLMRobertaConfig(**cfg)
            self.model = XLMRobertaForMaskedLM(architecture_config)

        # ToDo: For now, just inherit forward method
        forward = self.model.forward
