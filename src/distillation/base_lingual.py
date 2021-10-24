import pytorch_lightning as pl
from src.distillation.mixin.optimizer import OptimizerMixin
from src.distillation.mixin.eval import EvalMixin
import torch
from omegaconf import DictConfig, OmegaConf
from src.utils import utils
from src.models.model import initialize_teacher_or_student
import hydra
from src.utils.utils import get_subset_dict, get_language_subset_batch, keep_only_model_forward_arguments, \
    get_model_language, name_model_for_logger, append_torch_in_dict
from transformers.tokenization_utils_base import BatchEncoding
from src.datamodules.mixed_data import MixedDataModule

log = utils.get_logger(__name__)


class BaseLingual(OptimizerMixin, EvalMixin, pl.LightningModule):
    def __init__(
            self,
            train_cfg: DictConfig,
            teacher_cfg: DictConfig,
            student_cfg: DictConfig,
            data_cfg: DictConfig,
            evaluation_cfg: DictConfig,
            *args,
            **kwargs,
    ):

        pl.LightningModule.__init__(self)
        super().__init__()

        self.train_cfg = train_cfg
        self.teacher_cfg = teacher_cfg
        self.student_cfg = student_cfg
        self.evaluation_cfg = evaluation_cfg
        self.data_cfg = data_cfg

        # language_mapping is being initialized in mono/bi/multilingual class
        self.number_of_models = len(self.language_mapping["model_id"])
        self.languages = self.language_mapping["id_model"]

        self.model = []
        self.student_tokenizers = []
        # Initialize Student Model and corresponding tokenizer
        for i in range(self.number_of_models):
            tokenizer, model = initialize_teacher_or_student(self.student_cfg)
            self.student_tokenizers.append(tokenizer)
            self.model.append(model)

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg.model._target_}>")
        self.teacher_tokenizer, self._teacher = initialize_teacher_or_student(self.student_cfg)
        self._teacher.eval()
        self.teacher_outputs = None

        # Init Data Module
        log.info(f"Instantiating datamodule")
        if hasattr(self.data_cfg, "_target_"):
            self.datamodule = hydra.utils.instantiate(self.data_cfg,
                                                      language_mapping=self.language_mapping,
                                                      s_tokenizer=self.student_tokenizers,
                                                      t_tokenizer=self.teacher_tokenizer)

        else:
            # TODO: Maybe another structure? "Hardcoded" mixed data structure.
            self.datamodule = MixedDataModule(self.data_cfg,
                                              language_mapping=self.language_mapping,
                                              s_tokenizer=self.student_tokenizers,
                                              t_tokenizer=self.teacher_tokenizer)

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

        output = self.common_step(model_idx=optimizer_idx, batch=batch, prefix="train")

        for key, value in output["log"].items():
            self.log(key, value)

        return output

    def common_step(self, model_idx: int, batch: BatchEncoding, prefix: str):

        if model_idx == 0:
            # Calculate Teacher Outputs (don't need gradient)
            with torch.no_grad():
                full_batch = keep_only_model_forward_arguments(self._teacher,
                                                               batch,
                                                               remove_additional_keys=["labels"])

                self.teacher_outputs = self._teacher.forward(**full_batch)  # (bs, seq_length, voc_size)

        model_languages = get_model_language(model_idx, self.language_mapping)

        subset_batch, idx = get_language_subset_batch(batch,
                                                      self.language_mapping,
                                                      model_languages)

        cleaned_batch = keep_only_model_forward_arguments(self.model[model_idx],
                                                          subset_batch,
                                                          remove_additional_keys=["labels"])

        # Get corresponding Teacher Outputs
        subset_teacher_output = get_subset_dict(self.teacher_outputs, idx)

        # Get Loss
        student_outputs = self.model[model_idx](**cleaned_batch)  # (bs, seq_length, voc_size)
        loss = self.loss(student_outputs, subset_teacher_output, subset_batch["labels"])

        # TODO: Model ID
        model_name_logger = name_model_for_logger(model_languages)
        tqdm_dict = {model_name_logger + "/" + prefix + "/" + "_".join(model_languages) + '/train_loss': loss}

        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        return output

    def forward(self, batch: BatchEncoding):

        output = {}
        for model_idx in range(self.number_of_models):
            model_languages = get_model_language(model_idx, self.language_mapping)
            subset_batch, idx = get_language_subset_batch(batch,
                                                          self.language_mapping,
                                                          model_languages)
            cleaned_batch = keep_only_model_forward_arguments(self.model[model_idx],
                                                              subset_batch,
                                                              remove_additional_keys=["labels"])

            subset_output = self.model[model_idx].forward(**cleaned_batch)
            output[model_idx] = subset_output
            output[model_idx]["batch_idx"] = idx

        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        language_pair = self.datamodule.validation_language_mapping[dataloader_idx].split("_")
        models_cfg = [single_model for single_model in self.val_logger_names if single_model["dataset"] == "_".join(language_pair)]
        val_outputs = {}

        for model_cfg in models_cfg:
            logger_name = model_cfg["logger_name"]
            if logger_name not in model_cfg.keys():
                val_outputs[logger_name] = {}

            selected_language = model_cfg["selected_language"]
            model_id = model_cfg["model_id"]

            if model_cfg["eval_with"] != "":
                model_eval_with_id = model_cfg["eval_with"]
            else:
                model_eval_with_id = None
            for current_model_id in [model_id, model_eval_with_id]:
                model_languages = get_model_language(current_model_id, self.language_mapping)
                language_overlap = list(set(model_languages) & set(selected_language.split("_")))
                subset_batch, idx = get_language_subset_batch(batch,
                                                              self.language_mapping,
                                                              language_overlap)

                cleaned_batch = keep_only_model_forward_arguments(self.model[current_model_id],
                                                                  subset_batch)

                # TODO: Change Model ID
                self.evaluation = model_cfg["cfg"]
                self.metrics = self.evaluation.metrics

                output_step = self.eval_step(cleaned_batch,
                                             stage=logger_name,
                                             model_idx=current_model_id)
                val_outputs[logger_name] = append_torch_in_dict(output_step, val_outputs[logger_name])

        return val_outputs

    def validation_epoch_end(self, validation_step_outputs: list):
        outputs = {}

        for value in validation_step_outputs:
            output_step = []
            logger_name = list(value[0].keys())[0]
            for item_ in value:
                output_step.append(item_[logger_name])
            for cfg in self.val_logger_names:
                if cfg["logger_name"] == logger_name:
                    # TODO: Change Model ID
                    self.evaluation = cfg["cfg"]
                    self.metrics = self.evaluation.metrics
            self.eval_epoch_end(logger_name, output_step)

    """
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        all_output = {}
        for model_idx in range(self.number_of_models):
            model_languages = get_model_language(model_idx, self.language_mapping)
            model_name_logger = name_model_for_logger(str(model_idx), model_languages)
            for language in model_languages:
                # TODO: Change Model ID
                model_name = str(model_idx) + "_" + language

                subset_batch, idx = get_language_subset_batch(batch,
                                                              self.language_mapping,
                                                              model_languages)

                cleaned_batch = keep_only_model_forward_arguments(self.model[model_idx],
                                                                  subset_batch)

                # TODO: Change Model ID
                self.metrics = self.evaluation.metrics[model_name]
                output_step = self.eval_step(cleaned_batch,
                                             stage=model_name_logger + "/val/" + language,
                                             model_idx=model_idx)

                all_output[model_name] = output_step

        return all_output
    """
