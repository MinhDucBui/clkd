from torchmetrics import Metric
import torch.nn as nn
import torch


class Perplexity(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
        self.add_state("ce_total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        assert logits.shape[:2] == labels.shape
        ce = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.ce_total += ce
        self.total += (labels != -100).sum()

    def compute(self):
        # compute final result
        return torch.exp(self.ce_total / self.total)
