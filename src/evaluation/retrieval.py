from torchmetrics import Metric
import torch
from tqdm import tqdm


class MRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("cls", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(self, cls: torch.Tensor, labels: torch.Tensor):
        self.cls.append(cls)
        self.labels.append(labels)

    def compute(self):
        self.cls = torch.cat([x for _, x in sorted(zip(self.labels, self.cls), key=lambda pair: pair[0])], 0)
        num = self.cls.shape[0]
        self.cls /= self.cls.norm(2, dim=-1, keepdim=True)
        src_embeds = self.cls[: num // 2]
        trg_embeds = self.cls[num // 2:]
        # (1000, 1000)
        preds = src_embeds @ trg_embeds.T
        reciprocal_rank = get_reciprocal_rank(preds)
        # compute final result
        return reciprocal_rank.mean()


class BERTScoreMRR(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("last_hidden_representation", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(self, last_hidden_representation: torch.Tensor, labels: torch.Tensor):
        self.last_hidden_representation.append(last_hidden_representation)
        self.labels.append(labels)

    def compute(self):
        self.last_hidden_representation = [x for _, x in sorted(zip(self.labels, self.last_hidden_representation),
                                                                key=lambda pair: pair[0])]
        bert_score = compute_bertscore(self.last_hidden_representation)
        reciprocal_rank = get_reciprocal_rank(bert_score)
        # compute final result
        return reciprocal_rank.mean()


def compute_bertscore(last_hidden_representation):
    out, att_mask = concatenate_3d(last_hidden_representation)
    num = out.shape[0]
    src_embeds = out[: num // 2]
    trg_embeds = out[num // 2:]
    src_att = att_mask[: num // 2]
    trg_att = att_mask[num // 2:]
    bert_score = bertscore_pytorch({"last_hidden_state": src_embeds, "attention_mask": src_att},
                                   {"last_hidden_state": trg_embeds, "attention_mask": trg_att})
    return bert_score


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


def greedy_cos_idf(ref_embedding, hyp_embedding):
    ref_embedding = torch.unsqueeze(ref_embedding, 0)
    hyp_embedding = torch.unsqueeze(hyp_embedding, 0)

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]
    hyp_idf = torch.full(hyp_embedding.size()[:2], 1.0)
    ref_idf = torch.full(ref_embedding.size()[:2], 1.0)
    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    F = F.masked_fill(torch.isnan(F), 0.0)

    return P, R, F


def bertscore_pytorch(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor]):
    x_attn_mask: torch.Tensor = x["attention_mask"].clamp(0)  # (n_sentences, n_tokens)
    x_embeds: torch.Tensor = x["last_hidden_state"]  # (n_sentences, n_tokens, hidden_dim)
    y_attn_mask: torch.Tensor = y["attention_mask"].clamp(0)  # (n_sentences, n_tokens)
    y_embeds: torch.Tensor = y["last_hidden_state"]  # (n_sentences, n_tokens, hidden_dim)
    N, L, C = x_embeds.shape
    M = torch.full_like(
        torch.Tensor(N, N), fill_value=0, device=x_embeds.device
    )

    print("Calculate BERTScore")
    for i in tqdm(range(N)):
        x_embedding = x_embeds[i, :, :]
        x_mask = x_attn_mask[i, :]
        x_embedding = x_embedding[x_mask.bool(), :]
        x_embedding = x_embedding / torch.linalg.norm(x_embedding, ord=2, dim=-1, keepdim=True)
        for j in range(i, N):
            y_embedding = y_embeds[j, :, :]
            y_mask = y_attn_mask[j, :]
            y_embedding = y_embedding[y_mask.bool(), :]
            y_embedding = y_embedding / torch.linalg.norm(y_embedding, ord=2, dim=-1, keepdim=True)
            _, _, F = greedy_cos_idf(x_embedding, y_embedding)
            M[i, j] = F
    M = M + M.T - torch.diag(torch.diag(M))
    return M


def concatenate_3d(tensors: list[torch.Tensor], pad_id=-100):
    # (N sequences, L individual sequence length, C num classes -- typically)
    N, L, C = zip(*[tuple(x.shape) for x in tensors])
    out = torch.full_like(
        torch.Tensor(sum(N), max(L), max(C)), fill_value=pad_id, device=tensors[0].device
    )
    att_mask = torch.full_like(
        torch.Tensor(sum(N), max(L)), fill_value=0, device=tensors[0].device
    )
    start = 0
    for t in tensors:
        num, len_, hidden_dim = t.shape
        out[start: start + num, :len_, :] = t

        att_mask[start: start + num, :len_] = torch.full_like(
            torch.Tensor(num, len_), fill_value=1, device=t.device
        )
        start += num
    return out, att_mask
