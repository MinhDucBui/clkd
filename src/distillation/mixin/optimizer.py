import hydra
import sys
from src.utils import utils

log = utils.get_logger(__name__)


class OptimizerMixin:
    def setup(self, stage):
        if stage == "fit":
            if self.trainer_cfg.max_steps:
                self.total_steps = self.trainer_cfg.max_steps
                self.num_training_steps = self.trainer_cfg.max_steps
            else:
                sys.exit("As we are using an IterableDataset structure, please specify the max_steps in Trainer.")

    def base_configure_optimizers(self):
        optimizers = []
        lr_schedulers = []
        for i in range(self.number_of_models):
            model_name = self.student_mapping["id_model"][i]["model_name"]
            optimizer_cfg = self.students_model_cfg[model_name].optimizer
            optimizer = hydra.utils.instantiate(optimizer_cfg, self.get_transformer_params(model_idx=i))
            optimizers.append(optimizer)
            if 'lr_scheduler' in self.students_model_cfg[model_name]:
                lr_scheduler_cfg = self.students_model_cfg[model_name].lr_scheduler
                lr_scheduler = self.configure_scheduler(lr_scheduler_cfg, optimizer)
                lr_schedulers.append(lr_scheduler)
        return optimizers, lr_schedulers

    def base_configure_scheduler(self, lr_scheduler_cfg, optimizer):
        if hasattr(lr_scheduler_cfg, "num_warmup_steps") and isinstance(
                lr_scheduler_cfg.num_warmup_steps, float
        ):
            lr_scheduler_cfg.num_warmup_steps *= self.num_training_steps
            log.info(
                f"Warm up for {lr_scheduler_cfg.num_warmup_steps} of {self.num_training_steps}"
            )
            scheduler = hydra.utils.instantiate(lr_scheduler_cfg, optimizer, num_training_steps=self.num_training_steps)
        else:
            scheduler = hydra.utils.instantiate(lr_scheduler_cfg, optimizer)

        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    def configure_scheduler(self, lr_scheduler_cfg, optimizer):
        schedule = self.base_configure_scheduler(lr_scheduler_cfg, optimizer)
        return schedule

    def configure_optimizers(self):
        optimizers, lr_schedulers = self.base_configure_optimizers()
        return optimizers, lr_schedulers

    def get_transformer_params(self, model_idx):
        embedding_parameters = []
        for key, value in self.embeddings[model_idx].items():
            embedding_parameters += list(value.parameters())

        return list(self.model[model_idx].parameters()) + embedding_parameters

    def get_all_transformer_params(self):
        transformer_params = []
        for i in range(self.number_of_models):
            transformer_params += self.get_transformer_params(i)
        return transformer_params
