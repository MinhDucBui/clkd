"""
   Loss functions
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from src.utils import utils

log = utils.get_logger(__name__)


class LossHintonKD(nn.Module):  # custom loss_f should inherit from nn.Module
    """
    Loss with Hinton knowledge distillation.
    """

    def __init__(self, temperature, alpha):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def get_loss(self, student_outputs, teacher_outputs, labels):
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

    def forward(self, student_outputs, teacher_outputs, labels):
        return self.get_loss(student_outputs, teacher_outputs, labels)

    def __call__(self, student_outputs, teacher_outputs, labels):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():
           labels ():

        Returns:

        """

        return self.get_loss(student_outputs, teacher_outputs, labels)


class TinyBertLoss(LossHintonKD):
    def __init__(self, temperature, alpha, loss_regularization, mapping=None):
        super().__init__(temperature, alpha)
        self.temperature = temperature
        self.alpha = alpha
        self.emb_loss_param = loss_regularization.emb_loss
        self.hidden_loss_param = loss_regularization.hidden_loss
        self.att_loss_param = loss_regularization.att_loss
        self.kd_loss_param = loss_regularization.kd_loss
        self.layer_mapping = mapping

    def forward(self, student_outputs, teacher_outputs, labels):

        mapping_factor = int((len(teacher_outputs['hidden_states']) - 1) / (len(student_outputs['hidden_states']) - 1))

        # Assure that layers can be mapped
        assert (len(
            teacher_outputs[
                'hidden_states']) - 1) % mapping_factor == 0, "Not able to map teacher layers to student layers. Change the number of student's layers."

        if not self.layer_mapping:
            # Create mapping dict
            mapping_hid = {0: 0}
            for i in range(1, len(student_outputs['hidden_states'])):
                mapping_hid[i] = i * mapping_factor

            mapping_att = mapping_hid[1:]
            mapping_att = [(x - 1, y - 1) for (x, y) in mapping_att]

        else:
            mapping_hid = self.layer_mapping
            mapping_att = {key - 1: value - 1 for (key, value) in mapping_hid.items()}

        # Embedding loss
        emb_loss = F.mse_loss(teacher_outputs['hidden_states'][0], student_outputs['hidden_states'][0])

        # Hidden states & Attention losses
        hidden_loss = 0
        att_loss = 0

        for key, value in mapping_hid.items():
            hidden_loss += F.mse_loss(teacher_outputs['hidden_states'][value], student_outputs['hidden_states'][key])

        for key, value in mapping_att.items():
            # For faster convergence, very small attention scores are set to 0.
            att_teacher = torch.where(teacher_outputs['attentions'][value] <= 1e-2,
                                      torch.zeros_like(teacher_outputs['attentions'][value]),
                                      teacher_outputs['attentions'][value])

            att_student = torch.where(student_outputs['attentions'][key] <= 1e-2,
                                      torch.zeros_like(student_outputs['attentions'][key]),
                                      student_outputs['attentions'][key])

            att_loss += F.mse_loss(att_teacher, att_student)

        # Add KD loss
        kd_loss = self.get_loss(student_outputs, teacher_outputs, labels)

        loss = self.emb_loss_param * emb_loss \
               + self.hidden_loss_param * hidden_loss \
               + self.att_loss_param * att_loss \
               + self.kd_loss_param * kd_loss

        return loss

    def __call__(self, student_outputs, teacher_outputs, labels):
        return self.forward(student_outputs, teacher_outputs, labels)
