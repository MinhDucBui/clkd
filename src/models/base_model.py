from transformers import AutoModelForMaskedLM, PreTrainedModel
from pytorch_lightning import LightningModule
from src.models.modules.optimizer import OptimizerMixin
from pytorch_lightning.utilities.parsing import AttributeDict
from typing import Any
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils_base import BatchEncoding
import torch as nn
from src.utils import utils
from typing import Union
log = utils.get_logger(__name__)


class BaseModel(LightningModule):
    hparams: AttributeDict

    def __init__(
            self,
            pretrained_model_name_or_path: Union[bool, str],
            **kwargs: Any,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.model = None
        if pretrained_model_name_or_path:
            try:
                self.model: nn.Module = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
            except OSError:
                message = "No pretrained '{name}' found! Use cfg architecture (random weights).".format(
                    name=pretrained_model_name_or_path)
                log.info(message)

