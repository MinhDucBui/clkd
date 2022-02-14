"""
   Loss functions
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from src.utils import utils

log = utils.get_logger(__name__)


class LossHiddenMSE(nn.Module):
    def __init__(self, mapping, cls_token_weight=1.0):
        super().__init__()
        self.mapping = mapping
        self.cls_token_weight = cls_token_weight

    def forward(self, teacher_outputs, student_outputs):
        hidden_loss = 0
        for key, value in self.mapping.items():
            if self.cls_token_weight == 1.0:
                hidden_loss += F.mse_loss(teacher_outputs['hidden_states'][value],
                                          student_outputs['hidden_states'][key])
            else:
                hidden_loss_cls = F.mse_loss(teacher_outputs['hidden_states'][value][:, 0, :],
                                              student_outputs['hidden_states'][key][:, 0, :]) * self.cls_token_weight
                hidden_loss += F.mse_loss(teacher_outputs['hidden_states'][value][:, 1:, :],
                                          student_outputs['hidden_states'][key][:, 1:, :])
                hidden_loss += hidden_loss_cls

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
    def __init__(self, mapping, cls_token_weight=1.0):
        super().__init__()
        self.cls_token_weight = cls_token_weight
        self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
        self.mapping = mapping

    def forward(self, teacher_outputs, student_outputs):
        hidden_loss = 0
        for key, value in self.mapping.items():
            dim = student_outputs['hidden_states'][key].size(-1)
            teacher_hidden_state = teacher_outputs['hidden_states'][value].view(-1, dim)
            student_hidden_state = student_outputs['hidden_states'][key].view(-1, dim)
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
    
    
class LossAttMSE(nn.Module):
    def __init__(self, mapping, cls_token_weight=1.0):
        super().__init__()
        self.cls_token_weight = cls_token_weight
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
