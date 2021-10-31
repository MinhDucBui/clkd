import pytorch_lightning as pl
from src.distillation.mixin.optimizer import OptimizerMixin
from src.distillation.mixin.eval import EvalMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
from src.models.model import initialize_teacher_or_student
import hydra
from src.utils.utils import get_subset_dict, keep_only_model_forward_arguments, get_model_language, \
    name_model_for_logger, append_torch_in_dict, initialize_evaluation_cfg, get_subset_cleaned_batch
from src.utils.mappings import create_mapping
from transformers.tokenization_utils_base import BatchEncoding
from src.datamodules.mixed_data import MixedDataModule
import itertools

log = utils.get_logger(__name__)


class BaseModule(OptimizerMixin, EvalMixin, pl.LightningModule):

    def __init__(
            self,
            cfg: DictConfig,
            *args,
            **kwargs,
    ):

        self.save_hyperparameters()
        self.cfg = cfg
        self.teacher_cfg = cfg.teacher
        self.students_cfg = cfg.students
        self.individual_students_cfg = cfg.students.individual
        self.data_cfg = cfg.datamodule
        self.trainer_cfg = cfg.trainer
        self.students_model_cfg = {}
        for model_name, model_cfg in self.individual_students_cfg.items():
            if "student_" not in model_name:
                continue
            self.students_model_cfg[model_name] = model_cfg

        # Initialize Evaluation
        self.evaluation_cfg = self.students_cfg.evaluation
        self.evaluation_cfg = initialize_evaluation_cfg(self.evaluation_cfg)

        # Map language to id, student to languages and get validation tasks
        self.language_mapping, self.student_mapping, self.validation_mapping \
            = create_mapping(self.students_cfg)

        self.number_of_models = len(self.student_mapping["model_id"])

        pl.LightningModule.__init__(self)
        super().__init__()

        # Init Students
        self.model, self.student_tokenizers, self.loss = [], [], []
        self.initialize_student_components()

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg.model._target_}>")
        self.teacher_tokenizer, self._teacher, self.teacher_outputs = None, None, None
        self.initialize_teacher()

        # Init Data Module
        log.info(f"Instantiating datamodule")
        self.initialize_datamodule()

    def initialize_teacher(self):
        self.teacher_tokenizer, self._teacher = initialize_teacher_or_student(self.teacher_cfg)
        self._teacher.eval()
        self.teacher_outputs = None

    def initialize_student_components(self):
        for model_name, model_cfg in self.students_model_cfg.items():
            tokenizer, model = initialize_teacher_or_student(model_cfg)
            self.student_tokenizers.append(tokenizer)
            self.model.append(model)
            self.loss.append(hydra.utils.instantiate(model_cfg["loss"]))

    def initialize_datamodule(self):
        if hasattr(self.data_cfg, "_target_"):
            self.datamodule = hydra.utils.instantiate(self.data_cfg,
                                                      languages=list(self.language_mapping["id_lang"].values()),
                                                      language_mapping=self.language_mapping,
                                                      s_tokenizer=self.student_tokenizers,
                                                      t_tokenizer=self.teacher_tokenizer)

        else:
            val_languages = []
            for task_name, task_cfg in self.evaluation_cfg.items():
                for evaluate_with_tuple in task_cfg["evaluate_with"]:
                    val_languages.append([evaluate_model[1] for evaluate_model in evaluate_with_tuple])
            val_languages.sort()
            val_languages = list(k for k, _ in itertools.groupby(val_languages))
            self.datamodule = MixedDataModule(self.data_cfg,
                                              train_languages=list(self.language_mapping["id_lang"].values()),
                                              eval_cfg=self.evaluation_cfg,
                                              val_languages=val_languages,
                                              language_mapping=self.language_mapping,
                                              s_tokenizer=self.student_tokenizers,
                                              t_tokenizer=self.teacher_tokenizer)

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
        model_idx = optimizer_idx
        if model_idx == 0:
            # Calculate Teacher Outputs (don't need gradient)
            with torch.no_grad():
                full_batch = keep_only_model_forward_arguments(self._teacher,
                                                               batch,
                                                               remove_additional_keys=["labels"])

                self.teacher_outputs = self._teacher.forward(**full_batch)  # (bs, seq_length, voc_size)

        model_languages = get_model_language(model_idx, self.student_mapping)
        cleaned_batch, subset_batch, idx = get_subset_cleaned_batch(self.model[model_idx], model_languages, batch,
                                                                    self.language_mapping,
                                                                    remove_additional_keys=["labels"])

        # Get corresponding Teacher Outputs
        subset_teacher_output = get_subset_dict(self.teacher_outputs, idx)

        # Get Loss
        student_outputs = self.model[model_idx](**cleaned_batch)  # (bs, seq_length, voc_size)
        loss = self.loss[model_idx](student_outputs, subset_teacher_output, subset_batch["labels"])

        # TODO: Model ID
        model_name_logger = name_model_for_logger(model_languages)
        tqdm_dict = {model_name_logger + "/" + "train" + "/" + "_".join(model_languages) + '/train_loss': loss}

        output = {
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

        for key, value in output["log"].items():
            self.log(key, value)

        return output

    def forward(self, batch: BatchEncoding):
        output = {}
        for model_idx in range(self.number_of_models):
            model_languages = get_model_language(model_idx, self.language_mapping)
            cleaned_batch, subset_batch, idx = get_subset_cleaned_batch(self.model[model_idx], model_languages, batch,
                                                                        self.language_mapping,
                                                                        remove_additional_keys=["labels"])

            subset_output = self.model[model_idx].forward(**cleaned_batch)
            output[model_idx] = subset_output
            output[model_idx]["batch_idx"] = idx

        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        language_pair = self.datamodule.validation_dataset_mapping[dataloader_idx]["languages"].split("_")
        task_name = self.datamodule.validation_dataset_mapping[dataloader_idx]["task"]
        models_cfg = [single_model for single_model in self.validation_mapping
                      if single_model["dataset"] == language_pair and single_model["task_name"] == task_name]
        val_outputs = {}
        for model_cfg in models_cfg:
            logger_name = model_cfg["logger_name"]
            if logger_name not in model_cfg.keys():
                val_outputs[logger_name] = {}

            model_tuple = [model_cfg["model_idx"], model_cfg["current_language"]]

            if model_cfg["eval_with"] != "":
                model_eval_tuples = model_cfg["eval_with"]
                model_eval_tuples = [[eval_tuple[0], eval_tuple[1].split("_")] for eval_tuple in model_eval_tuples]
            else:
                model_eval_tuples = [None, None]

            for current_model_tuple in [model_tuple] + model_eval_tuples:
                cleaned_batch, _, _ = get_subset_cleaned_batch(self.model[current_model_tuple[0]],
                                                               current_model_tuple[1],
                                                               batch,
                                                               self.language_mapping,
                                                               remove_additional_keys=[])

                # TODO: Change Model ID
                self.evaluation = model_cfg["cfg"]
                self.metrics = self.evaluation.metrics
                output_step = self.eval_step(cleaned_batch,
                                             stage=logger_name,
                                             model_idx=current_model_tuple[0])
                val_outputs[logger_name] = append_torch_in_dict(output_step, val_outputs[logger_name])
        return val_outputs

    def validation_epoch_end(self, validation_step_outputs: list):
        for value in validation_step_outputs:
            output_step = []
            for logger_name in list(value[0].keys()):
                for item_ in value:
                    output_step.append(item_[logger_name])
                for cfg in self.validation_mapping:
                    if cfg["logger_name"] == logger_name:
                        # TODO: Change Model ID
                        self.evaluation = cfg["cfg"]
                        self.metrics = self.evaluation.metrics
                self.eval_epoch_end(logger_name, output_step)
