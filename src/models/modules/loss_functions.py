"""
   Loss functions
"""

import torch.nn as nn
import torch.nn.functional as F


def loss_fn_kd(student_outputs, teacher_outputs, labels, hparams):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    """
    alpha = hparams.alpha
    temperature = hparams.temperature

    # Calculate Hilton KD Loss
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs / temperature, dim=1),
                             F.softmax(teacher_outputs / temperature, dim=1)) * (alpha * temperature * temperature)

    # Calculate Masked Language Model Loss
    if alpha > 0.0:
        cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = cross_entropy_loss(student_outputs.view(-1, student_outputs.size(-1)), labels.view(-1))

        KD_loss += alpha * loss_mlm

    return KD_loss
