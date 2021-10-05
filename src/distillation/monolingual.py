from src.distillation.base_lingual import BaseLingual
from src.utils.utils import get_subset_from_batch
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl


class Monolingual(BaseLingual, pl.LightningModule):
    def __init__(
            self,
            train_cfg: DictConfig,
            teacher_cfg: DictConfig,
            student_cfg: DictConfig,
            language_mapping: dict,
            *args,
            **kwargs,
    ):

        self.save_hyperparameters()

        super().__init__(train_cfg, teacher_cfg, student_cfg, language_mapping)

    def forward(self, batch: dict, language: torch.Tensor):

        language_id = language[:, 0]
        permuted_output = {}
        permute_tensor = []
        for index, single_model in enumerate(self.model):

            idx = (language_id == torch.tensor(index)).nonzero()[:, 0]
            subset_batch = get_subset_from_batch(batch, idx)
            # As e.g. DistillBert does not support token_type_ids
            cleaned_batch = {key: value for (key, value) in subset_batch.items() if
                             key in single_model.forward.__code__.co_varnames}

            subset_output = single_model.forward(**cleaned_batch)
            if index == 0:
                permute_tensor = idx
                permuted_output = subset_output
            else:
                permute_tensor = torch.cat((permute_tensor, idx), 0)
                permuted_output = {key: torch.cat((value, subset_output[key]), 0) for key, value in
                                   permuted_output.items()}

        # Permute to get original format
        output = {key: value[permute_tensor, :] for key, value in permuted_output.items()}
        return output


