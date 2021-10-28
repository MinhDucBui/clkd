from typing import Dict, Any

import pytorch_lightning as pl
from src.distillation.modules.optimizer import OptimizerMixin
import torch
from omegaconf import DictConfig, OmegaConf
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

        self.model = []
        self.student_tokenizers = []
        # Initialize Student Model and corresponding tokenizer
        for i in range(self.number_of_models):
            self.student_tokenizers.append(hydra.utils.instantiate(self.student_cfg.tokenizer))
            OmegaConf.update(self.student_cfg.model, "cfg.vocab_size", self.student_tokenizers[i].vocab_size)
            self.model.append(hydra.utils.instantiate(self.student_cfg.model))

        super().__init__()

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg.model._target_}>")
        self.teacher_tokenizer = hydra.utils.instantiate(self.teacher_cfg.tokenizer)
        OmegaConf.update(self.teacher_cfg.model, "cfg.vocab_size", self.teacher_tokenizer.vocab_size)
        self._teacher = hydra.utils.instantiate(self.teacher_cfg.model)

        self._teacher.eval()
        self.teacher_outputs = None

        # Init Data Module
        log.info(f"Instantiating datamodule <{self.data_cfg._target_}>")
        self.datamodule = hydra.utils.instantiate(self.data_cfg,
                                                  language_mapping=self.language_mapping,
                                                  s_tokenizer=self.student_tokenizers,
                                                  t_tokenizer=self.teacher_tokenizer)

        # TODO: Init Metric
        # self.metric = hydra.utils.instantiate(train_cfg.metric)

        # Init Loss
        self.loss = hydra.utils.instantiate(train_cfg.loss)

    # Trainer: Loops through batches (batch_idx) and then loops through optimizers (optimizer_idx)
    # In our case: One optimizer corresponds to a model
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """Trainer: Loops through batches (batch_idx) and then loops through optimizers (optimizer_idx).
        In our case: One optimizer corresponds to a model.

        Args:
            batch:
            batch_idx:
            optimizer_idx:

        Returns:

        """
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
                # MaskedLMOutput --> OrderedDict
                self.teacher_outputs = self._teacher.forward(**cleaned_batch)  # (bs, seq_length, voc_size)

        # Get corresponding languages
        model_languages = self.language_mapping["id_model"][model_idx][0].split("_")

        # Each batch consists of samples from multiple languages, but each model corresponds to a subset of languages
        # Idea: Get only the samples that corresponds to the model's languages
        # Get row index of the corresponding samples in the batch
        idx = get_language_subset_index(self.language_mapping, batch_language, model_languages)

        # DEBUG:
        # print("------\n\n")
        # for name, param in self.model[model_idx].state_dict().items():
        #    if name == "bert.encoder.layer.1.output.LayerNorm.weight":
        #        print(name, param)

        # Get corresponding Batch
        subset_batch = get_subset_dict(cleaned_batch, idx)
        subset_labels = labels[idx]

        # Get corresponding Teacher Outputs and Labels
        subset_teacher_output = get_subset_dict(self.teacher_outputs, idx)

        student_outputs = self.model[model_idx](**subset_batch)  # (bs, seq_length, voc_size)

        # Get Loss
        loss = self.loss(student_outputs, subset_teacher_output, subset_labels)

        tqdm_dict = {"_".join(model_languages) + '_loss': loss}

        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        self.log("_".join(model_languages) + "_" + prefix + '_loss', loss)

        return output
