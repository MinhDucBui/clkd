"""
   Loss functions
"""

import torch.nn as nn
from torch.nn import functional as F



class LossHiltonKD:
    """
    Loss with Hilton knowledge distillation.
    """

    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_outputs, teacher_outputs, labels):
        # Adapted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/distillation/distiller.py

        # Get Logits
        teacher_logits = teacher_outputs["logits"]
        student_logits = student_outputs["logits"]

        # Calculate Hinton KD Loss
        kl = F.kl_div(F.log_softmax(student_logits / self.temperature, dim=-1),
                      F.softmax(teacher_logits / self.temperature, dim=-1), reduction='batchmean')
        loss_kd = kl * self.alpha * self.temperature ** 2

        # Calculate Masked Language Model Loss
        if self.alpha > 0.0:
            cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
            loss_mlm = cross_entropy_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

            loss_kd += (1 - self.alpha) * loss_mlm

        return loss_kd

    def __call__(self, student_outputs, teacher_outputs, labels):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():
           labels ():

        Returns:

        """

        # Get Logits
        teacher_logits = teacher_outputs["logits"]
        student_logits = student_outputs["logits"]

        # Calculate Hilton KD Loss
        kd_loss = nn.KLDivLoss()(F.log_softmax(student_logits / self.temperature, dim=1),
                                 F.softmax(teacher_logits / self.temperature, dim=1)) * (self.alpha * self.temperature * self.temperature)

        # Calculate Masked Language Model Loss
        if self.alpha > 0.0:
            cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
            loss_mlm = cross_entropy_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

            kd_loss += self.alpha * loss_mlm

        return kd_loss
