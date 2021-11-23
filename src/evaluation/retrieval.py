from torchmetrics import Metric
import torch


class MRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("reciprocal_rank", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor):
        reciprocal_rank = get_reciprocal_rank(preds)
        self.reciprocal_rank += reciprocal_rank.sum()
        self.total += preds.shape[0]

    def compute(self):
        # compute final result
        return self.reciprocal_rank / self.total


def get_reciprocal_rank(preds: torch.Tensor) -> torch.Tensor:
    """Compute MRR from row-aligned matrices of square query-document pairs.

    `mrr` is primarily intended for BLI or sentence-translation retrieval.

    Args:
        preds: square matrix of ranking scores

    Returns:
        torch.Tensor: mean reciprocal rank
    """
    N = preds.shape[0]
    if torch.cuda.is_available():
        torch_arange = torch.arange(N)[:, None].to(device='cuda')
    else:
        torch_arange = torch.arange(N)[:, None]
    rankings = preds.argsort(dim=-1, descending=True) == torch_arange
    reciprocal_rank = 1 / (1 + rankings.float().argmax(dim=-1))
    return reciprocal_rank
