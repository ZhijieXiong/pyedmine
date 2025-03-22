import torch
import math
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_


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
        scores_ = nn.functional.softmax(scores_, dim=-1)
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
    scores = nn.functional.softmax(scores, dim=-1)

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
    scores = nn.functional.softmax(scores, dim=-1)  # BS,8,seq_len,seq_len

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
        scores_b = nn.functional.softmax(scores_b, dim=-1)
        scores = torch.cat([scores_a, scores_b], dim=2)

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seq_len).to(device)
        # 第一行score置0
        scores = torch.cat([pad_zero, scores[:bs, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)

    return output, scores

class MultiHeadAttention4SimpleKT(nn.Module):
    def __init__(self, params, bias=True):
        super().__init__()

        self.params = params
        model_config = self.params["models_config"]["SimpleKT"]
        dim_model = model_config["dim_model"]
        dropout = model_config["dropout"]
        key_query_same = model_config["key_query_same"]

        self.value_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.key_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if not key_query_same:
            self.query_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.bias_projection = bias
        self.projection_out = nn.Linear(dim_model, dim_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        key_query_same = self.params["models_config"]["SimpleKT"]["key_query_same"]
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        if not key_query_same:
            nn.init.xavier_uniform_(self.query_linear.weight)

        if self.bias_projection:
            nn.init.constant_(self.key_linear.bias, 0.)
            nn.init.constant_(self.value_linear.bias, 0.)
            if key_query_same is False:
                nn.init.constant_(self.query_linear.bias, 0.)
            nn.init.constant_(self.projection_out.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        model_config = self.params["models_config"]["SimpleKT"]
        key_query_same = model_config["key_query_same"]
        num_head = model_config["num_head"]
        dim_model = model_config["dim_model"]
        dim_head = dim_model // num_head
        batch_size = q.size(0)

        k = self.key_linear(k).view(batch_size, -1, num_head, dim_head)
        if key_query_same:
            q = self.key_linear(q).view(batch_size, -1, num_head, dim_head)
        else:
            q = self.query_linear(q).view(batch_size, -1, num_head, dim_head)
        v = self.value_linear(v).view(batch_size, -1, num_head, dim_head)

        # transpose to get dimensions (batch_size * num_head * seq_len * dim_model)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention4simple_kt(q, k, v, dim_head, mask, self.dropout, zero_pad, device=self.params["device"])

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
        output = self.projection_out(concat)

        return output
    
class MultiHeadAttention4AKT(nn.Module):
    def __init__(self, params, bias=True):
        super(MultiHeadAttention4AKT, self).__init__()
        self.params = params

        model_config = self.params["models_config"]["AKT"]
        dim_model = model_config["dim_model"]
        key_query_same = model_config["key_query_same"]
        num_head = model_config["num_head"]
        dropout = model_config["dropout"]

        self.dim_model = dim_model
        self.dim_feature = dim_model // num_head
        self.num_head = num_head
        self.key_query_same = key_query_same

        self.value_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.key_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if not key_query_same:
            self.query_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(dim_model, dim_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(num_head, 1, 1))

        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        if self.key_query_same is False:
            nn.init.xavier_uniform_(self.query_linear.weight)

        if self.proj_bias:
            nn.init.constant_(self.key_linear.bias, 0.)
            nn.init.constant_(self.value_linear.bias, 0.)
            if self.key_query_same is False:
                nn.init.constant_(self.query_linear.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, question_difficulty_emb):
        batch_size = q.size(0)
        k = self.key_linear(k).view(batch_size, -1, self.num_head, self.dim_feature)
        if not self.key_query_same:
            q = self.query_linear(q).view(batch_size, -1, self.num_head, self.dim_feature)
        else:
            q = self.key_linear(q).view(batch_size, -1, self.num_head, self.dim_feature)
        v = self.value_linear(v).view(batch_size, -1, self.num_head, self.dim_feature)

        # transpose to get dimensions batch_size * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention4akt(q, k, v, self.dim_feature, mask, self.dropout, zero_pad, gammas, question_difficulty_emb,
                               device=self.params["device"])

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)
        output = self.out_proj(concat)

        return output
    

class MultiHeadAttention4SparseKT(nn.Module):
    def __init__(self, params, bias=True):
        super().__init__()
        self.params = params

        model_config = self.params["models_config"]["SparseKT"]
        dim_model = model_config["dim_model"]
        key_query_same = model_config["key_query_same"]
        num_head = model_config["num_head"]
        dropout = model_config["dropout"]
        dim_feature = dim_model // num_head

        self.dim_model = dim_model
        self.dim_feature = dim_feature
        self.num_head = num_head
        self.key_query_same = key_query_same

        self.v_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.k_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if key_query_same is False:
            self.q_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(dim_model, dim_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        key_query_same = self.params["models_config"]["SparseKT"]["key_query_same"]

        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not key_query_same:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if not key_query_same:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        model_config = self.params["models_config"]["SparseKT"]
        dim_model = model_config["dim_model"]
        key_query_same = model_config["key_query_same"]
        num_head = model_config["num_head"]
        k_index = model_config["k_index"]
        dim_feature = dim_model // num_head

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, num_head, dim_feature)
        if key_query_same is False:
            q = self.q_linear(q).view(bs, -1, num_head, dim_feature)
        else:
            q = self.k_linear(q).view(bs, -1, num_head, dim_feature)
        v = self.v_linear(v).view(bs, -1, num_head, dim_feature)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, attn_weights = attention4sparse_kt(
            q,
            k,
            v,
            dim_feature,
            mask,
            self.dropout,
            zero_pad,
            k_index,
            self.params["device"]
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.dim_model)

        output = self.out_proj(concat)

        return output, attn_weights