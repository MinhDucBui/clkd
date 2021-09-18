import torchmetrics
from pytorch_lightning import LightningModule
import torch
from .modules.get_model_architecture import get_student_model_architecture
from .modules import pooling
from .modules.optimizer import OptimizerMixin
from transformers import (
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}



class StudentModel(OptimizerMixin, LightningModule):
    def __init__(
        self,
        student_model_type: str,
        student_model_name_or_path: str,
        pool_fn: str = "mean",
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # instantiate random model
        self.model = get_student_model_architecture(self.hparams)
        self.pool_fn = getattr(pooling, self.hparams.pool_fn)

        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def get_model_output(self, batch):
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            #position_ids=batch.get("position_ids", None)
        )

        return output

    def get_embedding(
            self,
            hidden_states,
            attention_mask
    ) -> torch.Tensor:
        # Get Last Hidden Layer
        last_hidden_state = hidden_states[-1]

        # Pool the hidden states
        embeds = self.pool_fn(
            hidden_states=last_hidden_state, attention_mask=attention_mask
        )

        return embeds


    def forward(self, batch: dict) -> torch.Tensor:
        """Used for inference only.

        Args:
            batch ():

        Returns:

        """

        output = self.get_model_output(batch)

        return output
