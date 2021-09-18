from pytorch_lightning import LightningModule, Trainer
from transformers import AutoModel, get_linear_schedule_with_warmup, AdamW


class OptimizerMixin(LightningModule):
    def setup(self, stage):
        if stage == "fit":
            self.total_steps = self.hparams.total_steps

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        param_optimizer = list(self.student.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
        print(
            f"Warm up for {self.hparams.warmup_steps * self.total_steps} of {self.total_steps}"
        )
        if self.hparams.lr_linear_decay:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps * self.total_steps,
                num_training_steps=self.total_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        return [optimizer]
