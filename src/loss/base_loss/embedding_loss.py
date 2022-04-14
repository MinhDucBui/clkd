"""
   Loss functions
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from src.utils import utils

log = utils.get_logger(__name__)


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
