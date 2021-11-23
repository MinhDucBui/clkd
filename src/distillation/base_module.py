import copy

import pytorch_lightning as pl
from src.distillation.mixin.optimizer import OptimizerMixin
from src.distillation.mixin.eval import EvalMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
from src.models.model import initialize_teacher_or_student, initialize_embeddings, change_embedding_layer
import hydra
from src.utils.utils import keep_only_model_forward_arguments, get_model_language, compare_models, \
    name_model_for_logger, append_torch_in_dict, initialize_evaluation_cfg, get_subset_cleaned_batch
from src.utils.assert_functions import assert_functions
from src.utils.mappings import create_mapping
from transformers.tokenization_utils_base import BatchEncoding
from src.datamodules.mixed_data import MixedDataModule
import itertools
from src.utils.parameter_sharing import embedding_sharing, weight_sharing
from src.utils.debug import debug_embedding_updating

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
        self.embedding_sharing_cfg = cfg.students.embed_sharing
        self.weight_sharing_cfg = cfg.students.weight_sharing_across_students
        self.validation_epoch_index = 0

        # Initialize Evaluation
        self.evaluation_cfg = self.students_cfg.evaluation
        self.evaluation_cfg = initialize_evaluation_cfg(self.evaluation_cfg)

        # Sanity Check Config
        assert_functions(self.students_model_cfg, self.embedding_sharing_cfg, self.weight_sharing_cfg,
                         self.evaluation_cfg)

        # Map language to id, student to languages and get validation tasks
        self.language_mapping, self.student_mapping, self.validation_mapping \
            = create_mapping(self.students_cfg)

        self.number_of_models = len(self.student_mapping["model_id"])

        pl.LightningModule.__init__(self)
        super().__init__()

        # Init Students
        self.model, self.student_tokenizers, self.loss, self.embeddings = [], [], [], []
        self.initialize_student_components()

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg.model._target_}>")
        self.teacher_tokenizer, self._teacher, self.teacher_outputs = None, None, {}
        self.initialize_teacher()

        # Init Data Module
        log.info(f"Instantiating datamodule")
        self.initialize_datamodule()

    def initialize_teacher(self):
        self.teacher_tokenizer, self._teacher = initialize_teacher_or_student(self.teacher_cfg)
        if torch.cuda.is_available():
            self._teacher = self._teacher.to(device='cuda')
        self._teacher.eval()

    def initialize_student_components(self):
        for model_name, model_cfg in self.students_model_cfg.items():
            tokenizer, model = initialize_teacher_or_student(model_cfg)
            if torch.cuda.is_available():
                model = model.to(device='cuda')
            embeddings = initialize_embeddings(model_cfg)
            self.model.append(model)
            self.embeddings.append(embeddings)
            self.student_tokenizers.append(tokenizer)
            self.loss.append(hydra.utils.instantiate(model_cfg["loss"]))

        embedding_sharing(self.embeddings, self.embedding_sharing_cfg, self.student_mapping)
        weight_sharing(self.weight_sharing_cfg, self.model, self.student_mapping)

    def initialize_datamodule(self):
        if hasattr(self.data_cfg, "_target_"):
            self.datamodule = hydra.utils.instantiate(self.data_cfg,
                                                      languages=list(self.language_mapping["id_lang"].values()),
                                                      language_mapping=self.language_mapping,
                                                      s_tokenizer=self.student_tokenizers,
                                                      t_tokenizer=self.teacher_tokenizer)

        else:
            val_languages = []
            # Get all languages that are needed for validation
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

        Workflow:
        -> If first iteration, then get the teacher outputs for current batch and save them for next iterations
        -> Get the language of the model and get the corresponding samples that are in the model (student) language
        -> Get corresponding Teacher Outputs (in the current model language)
        -> Calculate Loss and Log

        Args:
            batch:
            batch_idx:
            optimizer_idx:

        Returns:

        """

        model_idx = optimizer_idx
        # If first iteration, then get the teacher outputs for current batch and save them for next iterations
        if model_idx == 0:
            for language, single_batch in batch.items():
                # Calculate Teacher Outputs (don't need gradient)
                with torch.no_grad():
                    full_batch = keep_only_model_forward_arguments(self._teacher,
                                                                   single_batch,
                                                                   remove_additional_keys=["labels"])
                    # MaskedLMOutput --> OrderedDict
                    self.teacher_outputs[language] = self._teacher.forward(**full_batch)  # (bs, seq_length, voc_size)

        abs_loss = 0
        # Get the language of the model and get the corresponding samples that are in the model (student) language
        model_languages = get_model_language(model_idx, self.student_mapping)

        for language, single_batch in batch.items():
            if language not in model_languages:
                continue

            # Get corresponding Teacher Outputs (in the current batch language)
            subset_teacher_output = self.teacher_outputs[language]

            full_batch = keep_only_model_forward_arguments(self.model[model_idx],
                                                           single_batch,
                                                           remove_additional_keys=["labels"])

            change_embedding_layer(self.model[model_idx], model_idx, self.embeddings, language)

            # DEBUG:
            # debug_embedding_updating(self.model, model_idx, batch_idx, self.test, self.test1, language)

            student_outputs = self.model[model_idx](**full_batch)  # (bs, seq_length, voc_size)

            # Calculate Loss and Log
            abs_loss += self.loss[model_idx](student_outputs, subset_teacher_output, single_batch["labels"])

        model_name_logger = name_model_for_logger(model_languages)
        tqdm_dict = {model_name_logger + "/" + "train" + "/" + "_".join(model_languages) + '/train_loss': abs_loss}
        output = {
            'loss': abs_loss,
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
        """Each validation step has multiple validation dataloaders (indexed with dataloader_idx).
        self.validation_mapping contains information about which model is being validated on which task and dataset
        with the corresponding evaluation instructions.

        Workflow:
        -> Get language and task for current validation set with index dataloader_idx
        -> Go into self.validation_mapping and get which model should be validated (given language and task from
        previous step) and get the evaluation instructions.
        -> Loop through all models that are being validated
            -> Check for "eval_with" (another model validated with the current model, e.g. retrieval task)
            -> Execute evaluation instructions for the model (and eval_with if it exists)

        Args:
            batch:
            batch_idx:
            dataloader_idx:

        Returns:

        """

        val_outputs = {}

        # Get language and task for current validation set with index dataloader_idx
        language_pair = self.datamodule.validation_dataset_mapping[dataloader_idx]["languages"].split("_")
        task_name = self.datamodule.validation_dataset_mapping[dataloader_idx]["task"]

        # Go into self.validation_mapping and get which model should be validated (given language and task from
        # previous step) and get the evaluation instructions.
        models_cfg = [single_model for single_model in self.validation_mapping
                      if single_model["dataset"] == language_pair and single_model["task_name"] == task_name]

        for model_cfg in models_cfg:

            logger_name = model_cfg["logger_name"]
            if logger_name not in model_cfg.keys():
                val_outputs[logger_name] = {}
            model_tuple = [model_cfg["model_idx"], model_cfg["current_language"]]

            # Check for "eval_with" (another model validated with the current model, e.g. retrieval task)
            model_eval_tuples = model_cfg["eval_with"]
            model_eval_tuples = [[eval_tuple[0], eval_tuple[1].split("_")] for eval_tuple in model_eval_tuples]
            # Execute evaluation instructions for the model (and eval_with if it exists)
            for current_model_tuple in [model_tuple] + model_eval_tuples:
                if current_model_tuple[0] == "teacher":
                    current_model = self._teacher
                else:
                    current_model = self.model[current_model_tuple[0]]

                cleaned_batch, _, _ = get_subset_cleaned_batch(current_model,
                                                               current_model_tuple[1],
                                                               batch,
                                                               self.language_mapping,
                                                               remove_additional_keys=[])
                if cleaned_batch["input_ids"].nelement() == 0:
                    continue

                # TODO: Hardcoded, retrieval only needs to be computed once...
                # TODO: Best case is that every task only needs to be coded once... Change to this behaviour
                self.evaluation = model_cfg["cfg"]
                self.metrics = self.evaluation.metrics
                output_step = self.eval_step(cleaned_batch,
                                             stage=logger_name,
                                             model=current_model)
                val_outputs[logger_name] = append_torch_in_dict(output_step, val_outputs[logger_name])
        return val_outputs

    def validation_epoch_end(self, validation_step_outputs: list):
        """Aggregate all validation step outputs and calculate metric (if epoch_end=True)

        Workflow:
        -> Loop through all validation step outputs. In each step, outputs for different metrics could be
           available. E.g. step one has perplexity for language hh and mn on the validation dataset (hh, mn)
            -> Loop through the outputs for each metric
                -> Get the corresponding evaluation instructions and execute them

        Args:
            validation_step_outputs:

        Returns:

        """

        for model_eval_cfg in self.validation_mapping:
            logger_name = model_eval_cfg["logger_name"]
            output_step = []

            # Loop through all validation step outputs. In each step, outputs for different metrics could be
            # available. E.g. step one has perplexity for language hh and mn on the validation dataset (hh, mn)
            for value in validation_step_outputs:
                if isinstance(value, list):
                    for item_ in value:
                        if logger_name in item_.keys():
                            output_step.append(item_[logger_name])
                elif isinstance(value, dict):
                    output_step.append(value[logger_name])

            # Get the corresponding evaluation instructions and execute them
            self.evaluation = model_eval_cfg["cfg"]
            self.metrics = self.evaluation.metrics
            self.eval_epoch_end(logger_name, output_step)

