import sys

from src.distillation.base_module import BaseModule
from omegaconf import DictConfig
from src.utils import utils
import hydra
import torch.nn as nn
import torch
from src.models.modules.pooling import mean
from src.models.model import change_embedding_layer
from src.utils.utils import keep_only_model_forward_arguments
log = utils.get_logger(__name__)


class AdversarialLearning(BaseModule):
    """Adversarial Learning. Currently only supports 2 languages (binary case).

    """

    def __init__(
            self,
            cfg: DictConfig,
            *args,
            **kwargs,
    ):
        
        
        super().__init__(cfg, *args, **kwargs)
        self.language_out = nn.Linear(self.model[0].base_model.config.hidden_size, 1, bias=True)
        self.language_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
    def configure_optimizers(self):
        optimizers, lr_schedulers = self.base_configure_optimizers()
        transformer_params = []
        for i in range(self.number_of_models):
            embedding_parameters = []
            for key, value in self.embeddings[i].items():
                embedding_parameters += list(value.parameters())
            transformer_params += list(self.model[i].parameters()) + embedding_parameters

        # Get only unique objects
        """ 
        seen = collections.OrderedDict()
        for obj in all_params:
            if id(obj) in seen:
                print(obj)
            seen[id(obj)] = obj
        all_params = list(seen.values())
        """

        for model in ["generator", "discriminator"]:
            if model == "discriminator":
                optimizing_params = self.language_out.parameters()
            elif model == "generator":
                optimizing_params = transformer_params
            optimizer = hydra.utils.instantiate(self.cfg["distillation_setup"][model]["optimizer"], optimizing_params)
            optimizers.append(optimizer)
            if 'lr_scheduler' not in self.cfg["distillation_setup"][model]:
                lr_schedulers.append(None)
            else:
                lr_scheduler_cfg = self.cfg["distillation_setup"][model].lr_scheduler
                lr_scheduler = self.configure_scheduler(lr_scheduler_cfg, optimizer)
                lr_schedulers.append(lr_scheduler)

        return optimizers, lr_schedulers

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        generator_idx, discriminator_idx = self.number_of_models, self.number_of_models+1

        if optimizer_idx == generator_idx:
            output = self.generator_step(batch, batch_idx)
        elif optimizer_idx == discriminator_idx:
            output = self.discriminator_step(batch, batch_idx)
        else:
            output = self.base_training_step(batch, batch_idx, optimizer_idx)

        return output

    def pred_language(self, single_batch, attention_mask, language):

        for model_name, model_cfg in self.student_mapping["model_id"].items():
            # TODO: No overlapping languages permitted
            if language in model_cfg["languages"]:
                model_idx = model_cfg["idx"]
                break

        full_batch = keep_only_model_forward_arguments(self.model[model_idx],
                                                       single_batch,
                                                       remove_additional_keys=["labels"])

        change_embedding_layer(self.model[model_idx], model_idx, self.embeddings, language)
        student_outputs = self.model[model_idx](**full_batch)  # (bs, seq_length, voc_size)

        last_hidden_state = student_outputs['hidden_states'][-1]
        mean_pooled = mean(hidden_states=last_hidden_state, attention_mask=attention_mask)  # [B x E]
        logits = self.language_out(mean_pooled)  # [B x C]
        return logits

    def generator_loss(self, logits, target):
        """ Computes generator loss (L_G)
        Args:
            logits: logit scores from language classification module
            target: language labels (EN: 1, NON-EN: 0)
        Returns:
            loss: scalar value
        """
        # map label to reverse order
        # TODO: Does only work for binary classification!!
        target = (target == 0).float()
        
        loss = self.language_criterion(input=logits, target=target)
        return loss

    def discriminator_loss(self, logits, target):
        """ Computes discriminator loss (L_G)
        Args:
            logits: logit scores from language classification module
            target: language labels (EN: 1, NON-EN: 0)
        Returns:
            loss: scalar value
        """
        target = target.float()
        loss = self.language_criterion(input=logits,
                                       target=target)
        return loss

    def generator_step(self, batch, batch_idx):
        logits, labels = self.get_labels_logits(batch, batch_idx)
        g_loss = self.generator_loss(logits=logits, target=labels)
        tqdm_dict = {"train" + "/" + 'generator_loss' + "/train_loss": g_loss}
        output = {
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        for key, value in output["log"].items():
            self.log(key, value)
        return output

    def discriminator_step(self, batch, batch_idx):
        logits, labels = self.get_labels_logits(batch, batch_idx)
        d_loss = self.discriminator_loss(logits=logits, target=labels)
        tqdm_dict = {"train" + "/" + 'discriminator_loss' + "/train_loss": d_loss}
        output = {
            'loss': d_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        for key, value in output["log"].items():
            self.log(key, value)

        return output

    def get_labels_logits(self, batch, batch_idx):
        language_logits = {}
        language_labels = {}
        for language, single_batch in batch.items():
            language_logits[language] = self.pred_language(single_batch,
                                                           attention_mask=single_batch["attention_mask"],
                                                           language=language)
            language_labels[language] = torch.full(language_logits[language].size(),
                                                   self.language_mapping["lang_id"][language])

        iter_number_languages = range(len(self.language_mapping["id_lang"]))
        logits = torch.stack([language_logits[self.language_mapping["id_lang"][idx]] for idx in iter_number_languages],
                             dim=-1)
        labels = torch.stack([language_labels[self.language_mapping["id_lang"][idx]] for idx in iter_number_languages],
                             dim=-1)
        labels = labels.to(logits.device)
        return logits, labels
