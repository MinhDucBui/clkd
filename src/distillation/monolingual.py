from src.distillation.base_lingual import BaseLingual
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.utils.utils import monolingual_parse_mapping_language


class Monolingual(BaseLingual, pl.LightningModule):
    def __init__(
            self,
            s_lang,
            t_lang,
            train_cfg: DictConfig,
            teacher_cfg: DictConfig,
            student_cfg: DictConfig,
            data_cfg: DictConfig,
            *args,
            **kwargs,
    ):

        self.save_hyperparameters()

        self.s_lang = s_lang
        self.t_lang = t_lang
        self.language_mapping = {}
        self.create_monolingual_language_mapping()

        super().__init__(train_cfg, teacher_cfg, student_cfg, data_cfg)

    def create_monolingual_language_mapping(self):
        s_mapping_id_lang, s_index = monolingual_parse_mapping_language(self.s_lang, prefix="src", index=0)
        t_mapping_id_lang, _ = monolingual_parse_mapping_language(self.t_lang, prefix="trg", index=s_index)
        s_mapping_id_lang.update(t_mapping_id_lang)
        self.language_mapping["id_lang"] = s_mapping_id_lang
        mapping_lang_id = {v[0]: [k, v[1]] for k, v in self.language_mapping["id_lang"].items()}
        self.language_mapping["lang_id"] = mapping_lang_id
        self.language_mapping["model_id"] = self.language_mapping["lang_id"]
        self.language_mapping["id_model"] = self.language_mapping["id_lang"]

    """
    def forward(self, batch: dict, language: torch.Tensor):

        permuted_output = {}
        permute_tensor = []
        for index, single_model in enumerate(self.model):

            idx = (language == torch.tensor(index)).nonzero()[:, 0]
            subset_batch = get_subset_dict(batch, idx)
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
    """


