import hydra
from src.utils.utils import get_model_language
from src.distillation.base_module import BaseModule
from omegaconf import DictConfig
from src.utils import utils
import torch.nn as nn
import torch
import torch.nn.functional as f
from torch.distributions import Beta
from src.models.modules.pooling import cls
log = utils.get_logger(__name__)


class HICTLMixup(BaseModule):
    """HICTL with MixUp as negative examples.

    """

    def __init__(
            self,
            cfg: DictConfig,
            *args,
            **kwargs,
    ):

        super().__init__(cfg, *args, **kwargs)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.beta_dist = Beta(cfg.distillation_setup.mixup.alpha, cfg.distillation_setup.mixup.beta)

    def configure_optimizers(self):
        optimizers, lr_schedulers = self.base_configure_optimizers()
        transformer_params = self.get_all_transformer_params()

        for model in ["sictl"]:
            if model == "sictl":
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
        sictl_idx = self.number_of_models
        if optimizer_idx == sictl_idx:
            output = self.sctl_step(batch, batch_idx)
        else:
            output = self.base_training_step(batch, batch_idx, optimizer_idx)

        return output

    def sctl_step(self, batch, batch_idx):
        cls_token = self.pred_cls_token(batch)
        # TODO: Only works for binary cases!
        source = self.language_mapping["id_lang"][0]
        target = self.language_mapping["id_lang"][1]
        negatives_src_trg = self.mixup(cls_token[source], cls_token[target])
        negatives_trg_src = self.mixup(cls_token[target], cls_token[source])

        # Loss
        loss_sctl_src_trg = self.loss_sctl(cls_token[source], cls_token[target], negatives_src_trg)
        loss_sctl_trg_src = self.loss_sctl(cls_token[source], cls_token[target], negatives_trg_src)
        batch_size = negatives_src_trg.size()[0]
        loss = 1/(2*batch_size) * (torch.sum(loss_sctl_src_trg) + torch.sum(loss_sctl_trg_src))

        return {"loss": loss}

    def pred_cls_token(self, batch):
        for language, single_batch in batch.items():
            for model_name, model_cfg in self.student_mapping["model_id"].items():
                # TODO: No overlapping languages permitted
                if language in model_cfg["languages"]:
                    model_idx = model_cfg["idx"]
                    break

            cls_token = {}
            model_languages = get_model_language(model_idx, self.student_mapping)
            for language, single_batch in batch.items():
                if language not in model_languages:
                    continue
                student_outputs = self.forward(single_batch, model_idx, language)
                last_hidden_state = student_outputs['hidden_states'][-1]
                cls_token[language] = cls(hidden_states=last_hidden_state)

        return cls_token

    def mixup(self, inst1, inst2):
        """Mixs up rows in inst1 and inst2.
        Args:
            inst1 (:obj: `torch.Tensor`):
                tensor for first instance
            inst2 (:obj: `torch.Tensor`):
                tensor for second instance
        Returns:
            inputs, labels (:obj:`torch.Tensor`):
                Mixed-up inputs
        Authors: Zhang et al.
        Affiliation: Facebook AI Research
        Original paper: https://arxiv.org/abs/1710.09412
        """
        x1 = inst1
        x2 = inst2

        N1, D = x1.shape
        N2 = x2.shape[0]

        if N1 != N2:
            x2 = x2.unsqueeze(1).repeat(1, N1, 1)

        # assumes first axis is batch size
        # (N, 1)
        lda = self.beta_dist.sample(x1.shape[:1]).unsqueeze(-1).to(x1.device)
        _lda = 1 - lda

        x_ = lda * x1 + _lda * x2

        if N1 != N2:
            x_ = x_.view(-1, D)

        return x_

    def loss_sctl(self, source, target, negatives):
        nce = 0

        # Normalize for cos similarity
        source = f.normalize(source, p=2, dim=1)
        target = f.normalize(target, p=2, dim=1)
        negatives = f.normalize(negatives, p=2, dim=1)

        # Similarity Scores for positives
        # Dot product "row-wise" with two 2-D matrices, e.g. [X[0]@Y[0].T, X[1]@Y[1].T, ...]
        batch_size, dim = source.size()[0], source.size()[1]
        source = source.reshape(batch_size, 1, dim)
        target = target.reshape(batch_size, dim, 1)
        sim_scores_positive = torch.matmul(source, target).squeeze(1)

        # Similarity Scores for negatives
        negatives = torch.transpose(negatives, 0, 1)
        sim_scores_negatives = torch.matmul(source, negatives).squeeze(1)

        # Calculate L_sctl for x_i
        sim_scores = torch.concat((sim_scores_positive, sim_scores_negatives), dim=1)
        softmax_scores = - self.lsoftmax(sim_scores)[:, 0]

        return softmax_scores
