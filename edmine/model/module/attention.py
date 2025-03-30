import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def attention4simple_kt(q, k, v, dim_head, mask, dropout, zero_pad, device="cpu"):
    # dim_head: 每一个head的dim
    # scores: (batch_size, num_head, seq_len, seq_len)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.tensor(dim_head).float().sqrt().to(device)
    batch_size, num_head, seq_len = scores.size(0), scores.size(1), scores.size(2)
    scores.masked_fill_(mask == 0, -1e32)
    # scores: (batch_size, num_head, seq_len, seq_len)
    scores = torch.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(batch_size, num_head, 1, seq_len).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

def attention4akt(q, k, v, dim_head, mask, dropout, zero_pad, gamma=None, pdiff=None, device="cpu"):
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.tensor(dim_head).float().sqrt().to(device)
    batch_size, num_head, seq_len = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seq_len).expand(seq_len, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        # batch_size, num_head, seq_len, seq_len
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distance_cumulative = torch.cumsum(scores_, dim=-1)
        distance_total = torch.sum(scores_, dim=-1, keepdim=True)

        # 1, 1, seq_len, seq_len 位置差值
        position_effect = torch.abs(x1 - x2)[None, None, :, :]
        # score <0 时，设置为0
        dist_scores = torch.clamp((distance_total - distance_cumulative) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
    gamma = -1. * m(gamma).unsqueeze(0)

    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    if pdiff is None:
        # 对应论文公式1中的新增部分
        total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        # 对应论文公式1中的新增部分
        total_effect = torch.clamp(torch.clamp((dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    # batch_size, num_head, seq_len, seq_len
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        pad_zero = torch.zeros(batch_size, num_head, 1, seq_len).to(device)
        # 第一行score置0
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output

def attention4sparse_kt(q, k, v, dim_head, mask, dropout, zero_pad, k_index, device="cpu"):
    # BS, 8, seq_len, seq_len
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim_head)
    bs, head, seq_len = scores.size(0), scores.size(1), scores.size(2)
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seq_len,seq_len

    # sorted_attention：只用top-k，因为从论文消融实验来看top-k效果更好，并且原代码默认使用top-k
    if k_index + 1 >= seq_len:
        scores = scores
    else:
        scores_a = scores[:, :, : k_index + 1, :]
        scores_b = scores[:, :, k_index + 1:, :].reshape(
            bs * head * (seq_len - k_index - 1), -1
        )
        sorted_scores, sorted_idx = torch.sort(scores_b, descending=True)
        scores_t = sorted_scores[:, k_index - 1: k_index].repeat(1, seq_len)
        scores_b = torch.where(
            scores_b - scores_t >= 0, scores_b, torch.tensor(-1e16, dtype=torch.float32, device=device)
        ).reshape(bs, head, seq_len - k_index - 1, -1)
        # BS,8,seq_len,seq_len
        scores_b = F.softmax(scores_b, dim=-1)
        scores = torch.cat([scores_a, scores_b], dim=2)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seq_len).to(device)
        # 第一行score置0
        scores = torch.cat([pad_zero, scores[:bs, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output, scores


def attention4clkt(q, k, v, d_k, mask, dropout, device, gamma=None, zero_pad=True):
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
    bs, head, seq_len = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seq_len).expand(seq_len, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float()

        distance_cum_scores = torch.cumsum(scores_, dim=-1)

        distance_total_scores = torch.sum(scores_, dim=-1, keepdim=True)

        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor)
        position_effect = position_effect.to(device)

        dist_scores = torch.clamp(
            (distance_total_scores - distance_cum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = torch.nn.Softplus()

    gamma = -1.0 * m(gamma).unsqueeze(0)

    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    attn_scores = scores

    if zero_pad:
        # mask为0，第一行score置0
        pad_zero = torch.zeros(bs, head, 1, seq_len).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_scores


def attention4DTransformer(q, k, v, mask, gamma=None, max_out=False):
    dim_head = k.size(-1)
    # attention score with scaled dot production
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.tensor(dim_head).float().sqrt().to(gamma.device)
    batch_size, num_head, seq_len, _ = scores.size()

    # temporal effect, i.e., time with exponential decay
    if gamma is not None:
        x1 = torch.arange(seq_len).float().expand(seq_len, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distance_cumulative = torch.cumsum(scores_, dim=-1)
            distance_total = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            # AKT论文中计算gamma_{t,t'}的公式
            dist_scores = torch.clamp(
                (distance_total - distance_cumulative) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
        # AKT论文中公式(1)
        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    # set to hard zero to avoid leakage
    scores = scores.masked_fill(mask == 0, 0)

    # max-out scores (batch_size, num_head, seq_len, seq_len)
    if max_out:
        # 关注
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale

    # calculate output
    output = torch.matmul(scores, v)
    return output, scores