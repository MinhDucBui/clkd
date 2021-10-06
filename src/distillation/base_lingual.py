import pytorch_lightning as pl
from src.distillation.modules.optimizer import OptimizerMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
import hydra
from collections import OrderedDict
from src.utils.utils import get_subset_dict
#log = utils.get_logger(__name__)


class BaseLingual(OptimizerMixin, pl.LightningModule):
    def __init__(
            self,
            train_cfg: DictConfig,
            teacher_cfg: DictConfig,
            student_cfg: DictConfig,
            data_cfg: DictConfig,
            *args,
            **kwargs,
    ):

        super().__init__()

        self.train_cfg = train_cfg
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.data_cfg = data_cfg

        # Init Data Module
        #log.info(f"Instantiating datamodule <{self.data_cfg._target_}>")
        self.datamodule = hydra.utils.instantiate(self.data_cfg, language_mapping=self.language_mapping)

        # Init Teacher Model
        #log.info(f"Instantiating Teacher model <{self.teacher_cfg._target_}>")
        self._teacher = hydra.utils.instantiate(self.teacher_cfg)
        self._teacher.eval()
        self.teacher_outputs = None

        # Initialize Student Models
        self.number_of_models = len(self.language_mapping["model_id"])
        self.model = [hydra.utils.instantiate(self.student_cfg) for i in range(self.number_of_models)]

        # TODO: Init Metric
        # self.metric = hydra.utils.instantiate(train_cfg.metric)

        # Init Loss
        self.loss = hydra.utils.instantiate(train_cfg.loss)

    def training_step(self, batch, batch_idx, optimizer_idx=0, prefix="train"):
        batch_language = batch["language"]
        labels = batch["labels"]
        print(optimizer_idx)
        cleaned_batch = {key: value for key, value in batch.items() if key not in ["language", "labels"]}
        if optimizer_idx == 0:
            # Calculate Teacher Outputs (don't need gradient)
            with torch.no_grad():
                self.teacher_outputs = self._teacher.forward(**cleaned_batch)  # (bs, seq_length, voc_size)

        # Get corresponding languages
        languages = self.language_mapping["id_model"][optimizer_idx][0]
        languages = languages.split("_")

        # Get row index of the corresponding samples in the batch
        idx = self.get_language_subset_index(batch_language, languages)

        # Get corresponding Student Number
        student_model = self.model[optimizer_idx]

        # Get corresponding Batch
        subset_batch = get_subset_dict(cleaned_batch, idx)
        subset_labels = labels[idx]

        # Get corresponding Teacher Outputs and Labels
        subset_teacher_output = get_subset_dict(self.teacher_outputs, idx)

        # Get Loss
        student_outputs = student_model(**subset_batch)  # (bs, seq_length, voc_size)
        loss = self.loss(student_outputs, subset_teacher_output, subset_labels)

        tqdm_dict = {"_".join(languages) + '_loss': loss}

        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        self.log("_".join(languages) + "_" + prefix + '_loss', loss)

        return output

    def get_language_subset_index(self, batch_language, languages):
        idx = None
        for index, single_language in enumerate(languages):
            language_id = self.language_mapping["lang_id"][single_language][0]
            subset_index = (batch_language == torch.tensor(language_id)).nonzero()[:, 0]
            if index == 0:
                idx = subset_index
            else:
                idx = torch.cat((idx, subset_index), 0)
        return idx

    """
    def common_step(self, student_model, batch: dict, batch_idx: int, prefix: str):
        # Prepare Batch
        language = batch["language"]
        labels = batch["labels"]
        batch = {key: value for key, value in batch.items() if key not in ["language", "labels"]}

        # Calculate Student Outputs
        student_outputs = self.student_model(batch, language)  # (bs, seq_length, voc_size)

        loss = self.loss(student_outputs, self.teacher_outputs, labels)

        # TODO: Add Metric Option
        # self.metric = self.metric(student_outputs["logits"], labels)
        # self.metric = {f'{prefix}_{k}': v for k, v in self.metric.items()}

        self.log(f'{prefix}_loss', loss)

        return loss
    
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
    """

