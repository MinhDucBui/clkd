"""
   Loss functions
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from src.utils import utils

log = utils.get_logger(__name__)


class LossMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, student_outputs, labels):
        student_logits = student_outputs["logits"]
        return self.cross_entropy_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

    def __call__(self, student_outputs, labels, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           labels ():

        Returns:

        """

        return self.forward(student_outputs, labels)


class LossSoftTargetsCE(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, teacher_outputs, student_outputs):

        # Get Logits
        teacher_logits = teacher_outputs["logits"]
        student_logits = student_outputs["logits"]
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Calculate Hinton KD Loss
        ce = (- teacher_soft * student_soft).mean()
        loss_ce = ce * self.temperature ** 2

        return loss_ce

    def __call__(self, teacher_outputs, student_outputs, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs)


class LossSoftTargetsKL(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, teacher_outputs, student_outputs):
        # From https://github.com/huggingface/pytorch-transformers/blob/master/examples/distillation/distiller.py

        # Get Logits
        teacher_logits = teacher_outputs["logits"]
        student_logits = student_outputs["logits"]

        # Calculate Hinton KD Loss
        kl = F.kl_div(F.log_softmax(student_logits / self.temperature, dim=-1),
                      F.softmax(teacher_logits / self.temperature, dim=-1), reduction='batchmean')
        loss_kd = kl * self.temperature ** 2
        return loss_kd

    def __call__(self, teacher_outputs, student_outputs, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs)


class LossAttMSE(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping
        self.mapping = {key - 1: value - 1 for (key, value) in self.mapping.items()}

    def forward(self, teacher_outputs, student_outputs):
        att_loss = 0
        for key, value in self.mapping.items():
            att_teacher = torch.where(teacher_outputs['attentions'][value] <= 1e-2,
                                      torch.zeros_like(teacher_outputs['attentions'][value]),
                                      teacher_outputs['attentions'][value])

            att_student = torch.where(student_outputs['attentions'][key] <= 1e-2,
                                      torch.zeros_like(student_outputs['attentions'][key]),
                                      student_outputs['attentions'][key])

            att_loss += F.mse_loss(att_teacher, att_student)

        return att_loss

    def __call__(self, teacher_outputs, student_outputs, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs)


class LossEmbeddingMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, teacher_outputs, student_outputs):
        # Embedding loss
        emb_loss = F.mse_loss(teacher_outputs['hidden_states'][0], student_outputs['hidden_states'][0])

        return emb_loss

    def __call__(self, teacher_outputs, student_outputs, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs)


class LossHiddenMSE(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def forward(self, teacher_outputs, student_outputs):
        hidden_loss = 0
        for key, value in self.mapping.items():
            hidden_loss += F.mse_loss(teacher_outputs['hidden_states'][value],
                                      student_outputs['hidden_states'][key])

        return hidden_loss

    def __call__(self, teacher_outputs, student_outputs, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs)


class LossHiddenCos(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
        self.mapping = mapping

    def forward(self, teacher_outputs, student_outputs):
        hidden_loss = 0
        for key, value in self.mapping.items():
            teacher_hidden_state = teacher_outputs['hidden_states'][value]
            student_hidden_state = student_outputs['hidden_states'][key]
            target = student_hidden_state.new(student_hidden_state.size(0)).fill_(1)

            # Embedding loss
            hidden_loss += self.cosine_loss_fct(teacher_hidden_state, student_hidden_state, target)

        return hidden_loss

    def __call__(self, teacher_outputs, student_outputs, *args, **kwargs):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs)
