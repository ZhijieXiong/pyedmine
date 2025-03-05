import torch
import torch.nn as nn


def attention(q, k, v, dim_head, mask, dropout, zero_pad, device="cpu"):
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
        scores = attention(q, k, v, dim_head, mask, self.dropout, zero_pad, device=self.params["device"])

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, dim_model)
        output = self.projection_out(concat)

        return output