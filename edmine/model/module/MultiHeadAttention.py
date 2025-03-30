import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_

from edmine.model.module.attention import *


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
    
    
class MultiHeadAttention4CLKT(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention4CLKT, self).__init__()
        self.params = params

        model_config = params["models_config"]["CLKT"]
        dim_model = model_config["dim_model"]
        num_head = model_config["num_head"]
        key_query_same = model_config["key_query_same"]
        dropout = model_config["dropout"]

        self.bias = True
        self.proj_bias = self.bias
        self.v_linear = nn.Linear(dim_model, dim_model, bias=self.bias)
        self.k_linear = nn.Linear(dim_model, dim_model, bias=self.bias)
        if not key_query_same:
            self.q_linear = nn.Linear(dim_model, dim_model, bias=self.bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=self.bias)
        self.gammas = nn.Parameter(torch.zeros(num_head, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        model_config = self.params["models_config"]["CLKT"]
        key_query_same = model_config["key_query_same"]

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

    def forward(self, q, k, v, mask, zero_pad=True):
        model_config = self.params["models_config"]["CLKT"]
        dim_model = model_config["dim_model"]
        num_head = model_config["num_head"]
        key_query_same = model_config["key_query_same"]
        dim_head = dim_model // num_head

        batch_size = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(batch_size, -1, num_head, dim_head)
        if key_query_same is False:
            q = self.q_linear(q).view(batch_size, -1, num_head, dim_head)
        else:
            q = self.k_linear(q).view(batch_size, -1, num_head, dim_head)
        v = self.v_linear(v).view(batch_size, -1, num_head, dim_head)

        # transpose to get dimensions batch_size * num_heads * seq_len * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores, attn_scores = attention4clkt(
            q, k, v, dim_head, mask, self.dropout, self.params["device"], self.gammas, zero_pad
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
        output = self.out_proj(concat)

        return output, attn_scores


class MultiHeadAttention4Dtransformer(nn.Module):
    def __init__(self, params, bias=True):
        super().__init__()
        self.params = params

        model_config = self.params["models_config"]["DTransformer"]
        dim_model = model_config["dim_model"]
        num_head = model_config["num_head"]
        key_query_same = model_config["key_query_same"]

        self.query_linear = nn.Linear(dim_model, dim_model, bias=bias)
        if key_query_same:
            self.key_linear = self.query_linear
        else:
            self.key_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.value_linear = nn.Linear(dim_model, dim_model, bias=bias)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(num_head, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, max_out=False):
        model_config = self.params["models_config"]["DTransformer"]
        dim_model = model_config["dim_model"]
        num_head = model_config["num_head"]
        dim_head = dim_model // num_head

        batch_size = q.size(0)
        # perform linear operation and split into num_head
        q = self.query_linear(q).view(batch_size, -1, num_head, dim_head)
        k = self.key_linear(k).view(batch_size, -1, num_head, dim_head)
        v = self.value_linear(v).view(batch_size, -1, num_head, dim_head)

        # transpose to get dimensions batch_size * num_head * seq_len * dim_head
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        v_, scores = attention4DTransformer(
            q,
            k,
            v,
            mask,
            self.gammas,
            max_out,
        )

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)

        output = self.out_proj(concat)

        return output, scores
