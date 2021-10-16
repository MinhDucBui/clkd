from pytorch_lightning import LightningModule, Trainer
import hydra
import sys
from src.utils import utils

log = utils.get_logger(__name__)


class OptimizerMixin:
    def setup(self, stage):
        if stage == "fit":
            if self.train_cfg.total_steps:
                self.total_steps = self.train_cfg.total_steps
                self.num_training_steps = self.train_cfg.total_steps
            else:
                sys.exit("As we are using an IterableDataset structure, please specify the max_steps in Trainer.")

    def configure_scheduler(self, optimizers):
        schedulers = []
        for i, optimizer in enumerate(optimizers):
            if i == 0:
                if hasattr(self.train_cfg.lr_scheduler, "num_warmup_steps") and isinstance(
                        self.train_cfg.lr_scheduler.num_warmup_steps, float
                ):
                    self.train_cfg.lr_scheduler.num_warmup_steps *= self.num_training_steps
                self.train_cfg.lr_scheduler.num_training_steps = self.num_training_steps
            log.info(
                f"Warm up for {self.train_cfg.lr_scheduler.num_warmup_steps} of {self.train_cfg.lr_scheduler}"
            )
            scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler, optimizer, )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            schedulers.append(scheduler)
        return schedulers

    def configure_optimizers(self):
        optimizers = []
        for i in range(self.number_of_models):
            optimizers.append(hydra.utils.instantiate(self.train_cfg.optimizer, self.model[i].parameters()))

        if 'lr_scheduler' not in self.train_cfg:
            return optimizers
        else:
            lr_schedulers = self.configure_scheduler(optimizers)
            return optimizers, lr_schedulers
