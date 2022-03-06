import copy
import pytorch_lightning as pl
from src.distillation.mixin.optimizer import OptimizerMixin
from src.distillation.mixin.eval import EvalMixin
import torch
from omegaconf import DictConfig
from src.utils import utils
from src.models.model import initialize_model
from src.models.modules.utils import change_embedding_layer
import hydra
from src.utils.utils import keep_only_model_forward_arguments, get_model_language, \
    append_torch_in_dict, initialize_evaluation_cfg, get_subset_cleaned_batch
from src.utils.assert_functions import assert_functions
from src.utils.mappings import create_mapping
from src.datamodules.mixed_data import MixedDataModule
import itertools
from src.utils.parameter_sharing import embedding_sharing, weight_sharing
from src.utils.debug import debug_embedding_updating
from src.utils.hydra import prepare_retrieval_eval
from src.models.modules.pooling import cls, mean

log = utils.get_logger(__name__)


class TeacherEval(OptimizerMixin, EvalMixin, pl.LightningModule):

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
        
        self.hidden_state_index = -1
        if "hidden_state_index" in cfg.keys():
            self.hidden_state_index = cfg.hidden_state_index

        # Initialize Evaluation
        self.evaluation_cfg = cfg.evaluation
        self.evaluation_cfg = initialize_evaluation_cfg(self.evaluation_cfg)

        # Map language to id, student to languages and get validation tasks
        self.language_mapping, self.student_mapping, self.validation_mapping, self.val_dataset_mapping \
            = create_mapping(self.students_cfg, self.evaluation_cfg, cfg.datamodule)

        self.number_of_models = len(self.student_mapping["model_id"])

        pl.LightningModule.__init__(self)
        super().__init__()

        # Init Teacher Model
        log.info(f"Instantiating Teacher model <{self.teacher_cfg.model._target_}>")
        self.teacher_tokenizer, self._teacher, self.teacher_outputs = None, None, {}
        self.initialize_teacher()
        
        # Init Students
        self.model, self.student_tokenizers, self.loss, self.embeddings = [], [], [], []
        self.initialize_student_components()
        
        self.a = torch.ones((2, 2), requires_grad=True)

    def initialize_teacher(self):
        self.teacher_tokenizer, self._teacher, _ = initialize_model(self.teacher_cfg)
        self._teacher.eval()

    def initialize_student_components(self):
        for model_name, model_cfg in self.students_model_cfg.items():
            tokenizer, model, embeddings = initialize_model(model_cfg, self._teacher)
            exec("self.%s = %s" % (model_name, "model"))
            for language, embedding in embeddings.items():
                exec("self.%s = %s" % (model_name + "_" + language, "embedding"))
            self.model.append(model)
            self.embeddings.append(embeddings)
            self.student_tokenizers.append(tokenizer)
            self.loss.append(hydra.utils.instantiate(model_cfg["loss"]))

        embedding_sharing(self.embeddings, self.embedding_sharing_cfg, self.student_mapping)
        weight_sharing(self.weight_sharing_cfg, self.model, self.student_mapping)


    # Trainer: Loops through batches (batch_idx) and then loops through optimizers (optimizer_idx)
    # In our case: One optimizer corresponds to a model
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return {"loss": self.a.mean()*0}


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
            model_eval_tuples = [[eval_tuple[0], eval_tuple[1].split("_")] for eval_tuple in model_eval_tuples]
            # Execute evaluation instructions for the model (and eval_with if it exists)
            current_model = self._teacher
            batch_language = self.language_mapping["id_lang"][batch["language"][0].item()]
            cleaned_batch = batch

            self.metrics = model_cfg["cfg"].metrics
            for k, v in self.metrics.items():
                outputs = current_model.forward(**{key: value for key, value in batch.items() 
                                                   if key not in ["labels", "language"]})

                logger_name = model_cfg["logger_name"]
                if k == "perplexity":
                    pass
                elif logger_name == "val/retrieval_cos_cls/student_turkish_tr-student_english_en":
                    kwargs = {}
                    kwargs["cls"] = cls(outputs.hidden_states[self.hidden_state_index], batch["attention_mask"])
                    kwargs["labels"] = batch["labels"]
                    v["metric"].update(**kwargs)
                elif logger_name == "val/retrieval_cos_mean/student_turkish_tr-student_english_en":
                    kwargs = {}
                    kwargs["cls"] = mean(outputs.hidden_states[self.hidden_state_index], batch["attention_mask"])
                    kwargs["labels"] = batch["labels"]
                    v["metric"].update(**kwargs)
                elif k == "bertscore_mrr":
                    kwargs = {}
                    kwargs["last_hidden_representation"] = outputs.hidden_states[self.hidden_state_index]
                    kwargs["labels"] = batch["labels"]
                    v["metric"].update(**kwargs)

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
    
        output_step = []

        for model_eval_cfg in self.validation_mapping:
            logger_name = model_eval_cfg["logger_name"]
            self.metrics = model_eval_cfg["cfg"].metrics
            for k, v in self.metrics.items():
                if k == "perplexity":
                    continue
                if getattr(v, "on_step", False):
                    self.log(f"{logger_name}/{k}", v["metric"].compute(), prog_bar=True)
                v["metric"].reset()
