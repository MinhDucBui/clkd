from pytorch_lightning import LightningModule, Trainer
import hydra
import sys


class OptimizerMixin:
    def setup(self, stage):
        if stage == "fit":
            if self.train_cfg.total_steps:
                self.total_steps = self.train_cfg.total_steps
            else:
                sys.exit("As we are using an IterableDataset structure, please specify the max_steps in Trainer.")

    def configure_optimizers(self):
        optimizers = []
        lr_schedulers = []
        for i in range(self.number_of_models):
            optimizers.append(hydra.utils.instantiate(self.train_cfg.optimizer, self.model[i].parameters()))

        if 'lr_scheduler' not in self.train_cfg:
            return optimizers
        else:
            for i in range(self.number_of_models):
                lr_schedulers.append(hydra.utils.instantiate(self.train_cfg.lr_scheduler, optimizer=optimizers[i],
                                                             num_training_steps=self.total_steps))
            return optimizers, lr_schedulers
