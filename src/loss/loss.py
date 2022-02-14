import torch.nn as nn
from src.utils import utils
from src.utils.assert_functions import assert_loss

log = utils.get_logger(__name__)


class GeneralLoss(nn.Module):
    def __init__(self, base_loss, loss_weighting):
        super().__init__()

        # Sanity Check CFG
        assert_loss(base_loss, loss_weighting)

        # Initialize
        self.base_loss = base_loss
        self.loss_weighting = loss_weighting


    def forward(self, teacher_outputs, student_outputs, labels):
        total_loss = 0
        for loss_name, loss in self.base_loss.items():
            if self.loss_weighting[loss_name] == 0:
                continue
            total_loss += self.loss_weighting[loss_name] * loss(teacher_outputs=teacher_outputs,
                                                                student_outputs=student_outputs,
                                                                labels=labels)

        return total_loss

    def __call__(self, teacher_outputs, student_outputs, labels):
        """ Compute the knowledge-distillation (KD) loss given outputs, labels.

        Args:
           student_outputs ():
           teacher_outputs ():
           labels ():

        Returns:

        """

        return self.forward(teacher_outputs, student_outputs, labels)
