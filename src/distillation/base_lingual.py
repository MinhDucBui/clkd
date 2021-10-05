import pytorch_lightning as pl
from src.distillation.modules.optimizer import OptimizerMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
import hydra
log = utils.get_logger(__name__)


class BaseLingual(OptimizerMixin, pl.LightningModule):
    def __init__(
            self,
            train_cfg: DictConfig,
            teacher_cfg: DictConfig,
            student_cfg: DictConfig,
            language_mapping: dict,
            *args,
            **kwargs,
    ):

        super().__init__()

        self.train_cfg = train_cfg
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.language_mapping = language_mapping

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{teacher_cfg._target_}>")
        self._teacher = hydra.utils.instantiate(teacher_cfg)
        self._teacher.eval()

        # Initialize Student Models
        self.number_of_models = len(self.language_mapping["model_id"])
        self.model = [hydra.utils.instantiate(self.student_cfg) for i in range(self.number_of_models)]

        # TODO: Init Metric
        # self.metric = hydra.utils.instantiate(train_cfg.metric)

        # Init Loss
        self.loss = hydra.utils.instantiate(train_cfg.loss)

    def common_step(self, batch: dict, batch_idx: int, prefix: str):
        # Prepare Batch
        language = batch["language"]
        labels = batch["labels"]
        batch = {key: value for key, value in batch.items() if key not in ["language", "labels"]}

        # Calculate Student Outputs
        student_outputs = self.forward(batch, language)  # (bs, seq_length, voc_size)

        # Calculate Teacher Outputs (don't need gradient)
        with torch.no_grad():
            teacher_outputs = self._teacher.forward(**batch)  # (bs, seq_length, voc_size)

        loss = self.loss(student_outputs, teacher_outputs, labels)

        # TODO: Add Metric Option
        # self.metric = self.metric(student_outputs["logits"], labels)
        # self.metric = {f'{prefix}_{k}': v for k, v in self.metric.items()}

        self.log(f'{prefix}_loss', loss)

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.common_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, prefix='val')

    # def validation_epoch_end(self, outputs):
    #    val_mse = self.val_mse.compute()

    #    TODO: Add Evaluation Metric
    #    self.log("val/mse", val_mse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, prefix='test')

    # def test_epoch_end(self, outputs):
    #    test_mse = self.test_mse.compute()

    #    TODO: Add Evaluation Metric
    #    self.log("test/mse", test_mse, prog_bar=True)

    # This has to be on_fit_start and not "setup" because "setup" doesn't have the right device
    def on_fit_start(self):
        # Since _teacher isn't a submodule, it's not automatically moved to the GPU device
        self._teacher.to(self.device, self.dtype)

    def on_test_epoch_start(self):
        self._teacher.to(self.device, self.dtype)