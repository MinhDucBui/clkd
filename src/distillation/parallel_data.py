from src.distillation.base_module import BaseModule
from omegaconf import DictConfig
from src.utils import utils
import hydra
import torch.nn as nn
import torch
from torch.nn import functional as F
from src.utils.utils import keep_only_model_forward_arguments, get_model_language, \
    append_torch_in_dict, get_subset_cleaned_batch

log = utils.get_logger(__name__)


class ParallelDataDistillation(BaseModule):
    """Adversarial Learning. Currently only supports 2 languages (binary case).

    """

    def __init__(
            self,
            cfg: DictConfig,
            *args,
            **kwargs,
    ):

        self.loss_weights = cfg.distillation_setup.loss_weights
        super().__init__(cfg, *args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx == 0:
            self.teacher_collect_outputs(batch, batch_idx)

        model_idx = optimizer_idx
        base_output = self.monodata_training(batch, model_idx)
        parallel_loss = self.parallel_data_training(batch, model_idx)
        abs_loss = base_output * self.loss_weights["mono_loss"] + parallel_loss * self.loss_weights["parallel_loss"]
        output = self.base_logging(model_idx, abs_loss)
        return output

    def parallel_data_training(self, batch, model_idx):

        model_languages = get_model_language(model_idx, self.student_mapping)
        for language, single_batch in batch.items():
            if len(language.split("-")) != 2:
                continue
            src_lang, trg_lang = language.split("-")[0], language.split("-")[1]
            if src_lang in model_languages:
                src_language_batch = batch[src_lang]
                src_outputs = self.forward(src_language_batch, model_idx, src_lang)
                # Look for target model
                trg_outputs = self.find_parallel_model_and_calculate_output(batch, model_idx, src_lang, trg_lang)
                break
            elif trg_lang in model_languages:
                trg_language_batch = batch[trg_lang]
                trg_outputs = self.forward(trg_language_batch, model_idx, trg_lang)
                src_outputs = self.find_parallel_model_and_calculate_output(batch, model_idx, trg_lang, src_lang)
                break
        parallel_loss = F.mse_loss(src_outputs['hidden_states'][-1], trg_outputs['hidden_states'][-1])
        return parallel_loss

    def find_parallel_model_and_calculate_output(self, batch, src_model_idx, src_lang, trg_lang):
        for model_name, model_cfg in self.student_mapping["model_id"].items():
            if trg_lang in model_cfg["languages"]:
                trg_language_batch = batch[src_lang]
                trg_model_idx = model_cfg["idx"]
                if trg_model_idx == src_model_idx:
                    trg_outputs = self.forward(trg_language_batch, trg_model_idx, trg_lang)
                else:
                    with torch.no_grad():
                        trg_outputs = self.forward(trg_language_batch, trg_model_idx, trg_lang)
                # Here we assume we only have one target model
                break
        return trg_outputs
