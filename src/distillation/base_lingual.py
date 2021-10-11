import pytorch_lightning as pl
from src.distillation.modules.optimizer import OptimizerMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
import hydra
from src.utils.utils import get_subset_dict, get_language_subset_index
log = utils.get_logger(__name__)


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

        self.train_cfg = train_cfg
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.data_cfg = data_cfg

        # language_mapping is being initialized in mono/bi/multilingual class
        self.number_of_models = len(self.language_mapping["model_id"])
        self.languages = self.language_mapping["id_model"]
        self.model = [hydra.utils.instantiate(self.student_cfg) for i in range(self.number_of_models)]

        super().__init__()

        # Init Data Module
        log.info(f"Instantiating datamodule <{self.data_cfg._target_}>")
        self.datamodule = hydra.utils.instantiate(self.data_cfg, language_mapping=self.language_mapping)

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg._target_}>")
        self._teacher = hydra.utils.instantiate(self.teacher_cfg)
        self._teacher.eval()
        self.teacher_outputs = None

        # TODO: Init Metric
        # self.metric = hydra.utils.instantiate(train_cfg.metric)

        # Init Loss
        self.loss = hydra.utils.instantiate(train_cfg.loss)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.common_step(model_idx=optimizer_idx, batch=batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        all_output = {}
        for model_idx in range(self.number_of_models):
            model_language = self.languages[model_idx][0].split("_")
            output = self.common_step(model_idx=model_idx, batch=batch, prefix="val")
            for key, value in output.items():
                output[key + "_" + "_".join(model_language)] = output.pop(key)
            all_output.update(output)
        return all_output

    def test_step(self, batch, batch_idx):
        all_output = {}
        for model_idx in range(self.number_of_models):
            model_language = self.languages[model_idx][0].split("_")
            output = self.common_step(model_idx=model_idx, batch=batch, prefix="test")
            for key, value in output.items():
                output[key + "_" + "_".join(model_language)] = output.pop(key)
            all_output.update(output)
        return all_output

    def common_step(self, model_idx: int, batch: dict, prefix: str):

        batch_language = batch["language"]
        labels = batch["labels"]
        cleaned_batch = {key: value for key, value in batch.items() if key not in ["language", "labels"]}
        # https://github.com/huggingface/transformers/issues/2702
        cleaned_batch = {key: value for (key, value) in cleaned_batch.items() if
                         key in self.model[model_idx].forward.__code__.co_varnames}
        if model_idx == 0:
            # Calculate Teacher Outputs (don't need gradient)
            with torch.no_grad():
                self.teacher_outputs = self._teacher.forward(**cleaned_batch)  # (bs, seq_length, voc_size)

        # Get corresponding languages
        model_languages = self.language_mapping["id_model"][model_idx][0].split("_")

        # Get row index of the corresponding samples in the batch
        idx = get_language_subset_index(self.language_mapping, batch_language, model_languages)

        # DEBUG:
        #print("------\n\n")
        #for name, param in self.model[model_idx].state_dict().items():
        #    if name == "bert.encoder.layer.1.output.LayerNorm.weight":
        #        print(name, param)

        # Get corresponding Batch
        subset_batch = get_subset_dict(cleaned_batch, idx)
        subset_labels = labels[idx]

        # Get corresponding Teacher Outputs and Labels
        subset_teacher_output = get_subset_dict(self.teacher_outputs, idx)

        # Get Loss
        student_outputs = self.model[model_idx](**subset_batch)  # (bs, seq_length, voc_size)
        loss = self.loss(student_outputs, subset_teacher_output, subset_labels)

        tqdm_dict = {"_".join(model_languages) + '_loss': loss}

        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        self.log("_".join(model_languages) + "_" + prefix + '_loss', loss)

        return output

    """
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

