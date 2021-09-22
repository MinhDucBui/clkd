from pytorch_lightning import LightningModule
from .modules.get_model_architecture import get_teacher_model_architecture
import torch
from .modules import pooling

class TeacherModel(LightningModule):
    def __init__(
        self,
        teacher_model_name_or_path: str,
        pool_fn: str = "mean",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # instantiate random model
        self.model = get_teacher_model_architecture(self.hparams)

        self.pool_fn = getattr(pooling, self.hparams.pool_fn)

    def get_model_output(self, batch):

        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            #position_ids=batch.get("position_ids", None),
            output_hidden_states=True
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

