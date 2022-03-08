import copy
import pytorch_lightning as pl
from src.distillation.mixin.optimizer import OptimizerMixin
from src.distillation.mixin.eval import EvalMixin
from src.distillation.mixin.initialize_models import InitializeModelsMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
from src.models.modules.utils import change_embedding_layer
from src.utils.utils import keep_only_model_forward_arguments, get_model_language, \
    append_torch_in_dict, get_subset_cleaned_batch
from src.utils.assert_functions import assert_functions
from src.utils.debug import debug_embedding_updating

log = utils.get_logger(__name__)


class BaseModule(OptimizerMixin, EvalMixin, InitializeModelsMixin, pl.LightningModule):

    def __init__(
            self,
            cfg: DictConfig,
            *args,
            **kwargs,
    ):

        # Sanity Check Config
        assert_functions(copy.deepcopy(cfg))

        self.save_hyperparameters()
        self.cfg = cfg
        self.data_cfg = cfg.datamodule
        self.trainer_cfg = cfg.trainer

        pl.LightningModule.__init__(self)
        super().__init__()

    def teacher_collect_outputs(self, batch, batch_idx) -> None:
        """ Collects teacher outputs from forward pass"""
        for language, single_batch in batch.items():
            # Calculate Teacher Outputs (don't need gradient)
            with torch.no_grad():
                full_batch = keep_only_model_forward_arguments(self._teacher,
                                                               single_batch,
                                                               remove_additional_keys=["labels"])
                # MaskedLMOutput --> OrderedDict
                self.teacher_outputs[language] = self._teacher.forward(**full_batch)  # (bs, seq_length, voc_size)

    def student_training_step(self, batch, batch_idx, optimizer_idx: int):
        """
        Defines pipeline for single student training:
            --> Collect languages of student
            --> For each language get corresponding teacher outputs
            --> Calculate loss for all languages of a student model

        @param batch: dict
        @param batch_idx: int
        @param optimizer_idx: int
        @return: dict
        """

        model_idx = optimizer_idx
        model_languages = get_model_language(model_idx, self.student_mapping)
        # model_languages: All languages that model contains, for monolingual models, there will be one language, for bilingual two, etc.
        abs_loss = 0

        # Iterate over those languages to get corresponding outputs from teacher
        assert len(self.teacher_outputs) > 0

        for language in model_languages:
            # Get teacher outputs for a single language
            subset_teacher_output = self.teacher_outputs[language]
            # Get batch item corresponding to this language
            language_batch = batch[language]
            # Get student outputs for corresponding language from the batch
            student_outputs = self.forward(language_batch, model_idx, language)

            # Assert that batch contains labels, they are needed for student MLM loss computation
            assert "labels" in language_batch
            abs_loss += self.loss[model_idx](subset_teacher_output, student_outputs, labels=language_batch["labels"])

        model_name = self.student_mapping["id_model"][model_idx]["model_name"]
        tqdm_dict = {"train" + "/" + model_name + '/train_loss': abs_loss}
        output = {
            'loss': abs_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        for key, value in output["log"].items():
            self.log(key, value)

        return output

    def base_training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx == 0:
            self.teacher_collect_outputs(batch, batch_idx)

        output = self.student_training_step(batch, batch_idx, optimizer_idx)
        return output

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """"
        Defines training loop of the model. Pipeline:
            --> First collect teacher outputs for a given batch
            --> For a given batch, loop over all student models (optimizer corresponds to a model) and compute training pipeline for a student
        """
        return self.base_training_step(batch, batch_idx, optimizer_idx)

    def forward(self, batch, model_idx, language):
        full_batch = keep_only_model_forward_arguments(self.model[model_idx],
                                                       batch,
                                                       remove_additional_keys=["labels"])

        change_embedding_layer(self.model[model_idx], model_idx, self.embeddings, language)

        # DEBUG:
        # debug_embedding_updating(self.model, model_idx, batch_idx, self.test, self.test1, language)
        output = self.model[model_idx](**full_batch)

        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Each validation step has multiple validation dataloader (indexed with dataloader_idx).
        self.validation_mapping contains information about which model is being validated on which task and dataset
        with the corresponding evaluation instructions.

        Workflow:

        * Get language and task for current validation set with index dataloader_idx
        * Go into self.validation_mapping and get which model should be validated (given language and task from previous step) and get the evaluation instructions.
        * Loop through all models that are being validated
            * Check for "eval_with" (another model validated with the current model, e.g. retrieval task)
            * Execute evaluation instructions for the model (and eval_with if it exists)

        Args:
            batch:
            batch_idx:
            dataloader_idx:

        Returns:

        """

        val_outputs = {}

        evaluation_name = self.val_dataset_mapping[dataloader_idx]

        # Go into self.validation_mapping and get which model should be validated (given language and task from
        # previous step) and get the evaluation instructions.
        # Order of languages does not matter
        models_cfg = [single_model for single_model in self.validation_mapping
                      if single_model["datamodule"] == evaluation_name]

        for model_cfg in models_cfg:

            logger_name = model_cfg["logger_name"]
            if logger_name not in model_cfg.keys():
                val_outputs[logger_name] = {}
            model_tuple = [model_cfg["model_idx"], model_cfg["current_language"]]

            # Check for "eval_with" (another model validated with the current model, e.g. retrieval task)
            model_eval_tuples = model_cfg["eval_with"]
            model_eval_tuples = [[eval_tuple[0], eval_tuple[1].split("_")] for eval_tuple in
                                 model_eval_tuples]  # ??? why this one is needed?

            # Execute evaluation instructions for the model (and eval_with if it exists)
            for current_model_tuple in [model_tuple] + model_eval_tuples:
                if current_model_tuple[0] == "teacher":
                    current_model = self._teacher
                else:
                    current_model = self.model[current_model_tuple[0]]

                batch_language = self.language_mapping["id_lang"][batch["language"][0].item()]

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
                                             model=current_model,
                                             language=batch_language,
                                             model_idx=current_model_tuple[0])
                if output_step:
                    val_outputs[logger_name] = append_torch_in_dict(output_step, val_outputs[logger_name])

        return val_outputs

    def validation_epoch_end(self, validation_step_outputs: list):
        """Aggregate all validation step outputs and calculate metric (if epoch_end=True)

        Workflow:

        * Loop through all validation step outputs. In each step, outputs for different metrics could be available. E.g. step one has perplexity for language hh and mn on the validation dataset (hh, mn)
            * Loop through the outputs for each metric
                * Get the corresponding evaluation instructions and execute them

        Args:
            validation_step_outputs:

        Returns:

        """

        """
        for name, param in self.model[0].named_parameters():
            if param.requires_grad:
                if "base.bert.encoder.layer.3.intermediate.dense.weight" == name:
                    print(name, param.data)
                if "base.cls.predictions.transform.LayerNorm.bias" == name:
                    print(name, param.data)
        for l, emb in self.embeddings[0].items():
            for name, param in emb.named_parameters():
                if param.requires_grad:
                    print(name, param.data)
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
