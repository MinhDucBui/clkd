"""
   Loss functions
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from src.utils import utils

log = utils.get_logger(__name__)


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

