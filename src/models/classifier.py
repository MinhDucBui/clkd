from typing import Optional

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from transformers import AutoModel

from .modules import pooling
from .modules.optimizer import OptimizerMixin
from .modules.heads import ClassificationHead

class SeqClassifier(OptimizerMixin, LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        lr: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: float = 0.1,
        weight_decay: float = 0.0,
        gpus: int = 1,
        accumulate_grad_batches: int = 1,
        max_epochs: int = 1,
        pool_fn: str = "mean",
        scl: bool = True,
        tau: float = 0.3,
        lr_linear_decay: bool = True,
        return_probs: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        # instantiate model
        self.model = AutoModel.from_pretrained(self.hparams.model_name_or_path)
        self.pool_fn = getattr(pooling, self.hparams.pool_fn)
        self.hparams.emb_dim = self.model.get_input_embeddings().embedding_dim
        self.head = ClassificationHead(self.hparams.emb_dim, self.hparams.num_labels)
        self.train()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state
        embeds = self.pool_fn(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        return embeds

    def forward(self, batch: dict) -> torch.Tensor:
        embeds = self.embed(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch.get("position_ids", None),
        )
        return embeds

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        embeds = self.embed(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch.get("position_ids", None),
        )
        loss = self.head(embeds, batch["labels"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        embeds = self.embed(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch.get("position_ids", None),
        )
        logits = self.head.linear(embeds).view(-1, self.hparams.num_labels)
        preds = logits.argmax(1)
        self.val_accuracy(preds, batch["labels"])

    def validation_epoch_end(self, outputs):
        val_acc = self.val_accuracy.compute()
        self.log("val/acc", val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        embeds = self.embed(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch.get("position_ids", None),
        )
        logits = self.head.linear(embeds).view(-1, self.hparams.num_labels)
        preds = logits.argmax(1)
        self.test_accuracy(preds, batch["labels"])

    def test_epoch_end(self, outputs):
        test_acc = self.test_accuracy.compute()
        self.log("test/acc", test_acc)
