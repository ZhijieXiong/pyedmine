import torch
import numpy as np
import torch.nn as nn

from edmine.model.module.MultiHeadAttention import *


class TransformerLayer4SimpleKT(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        model_config = self.params["models_config"]["SimpleKT"]
        dim_model = model_config["dim_model"]
        dim_ff = model_config["dim_ff"]
        dropout = model_config["dropout"]

        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention4SimpleKT(params)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        batch_size, seq_len = query.size(0), query.size(1)
        # 上三角和对角为1，其余为0的矩阵
        upper_triangle_ones = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype('uint8')
        # 用于取矩阵下三角
        src_mask = (torch.from_numpy(upper_triangle_ones) == 0).to(self.params["device"])
        if mask == 0:
            # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)
        # 残差
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query
    
    
class TransformerLayer4AKT(nn.Module):
    def __init__(self, params):
        super(TransformerLayer4AKT, self).__init__()
        self.params = params

        model_config = self.params["models_config"]["AKT"]
        dim_model = model_config["dim_model"]
        dim_ff = model_config["dim_ff"]
        dropout = model_config["dropout"]

        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention4AKT(params)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value, question_difficulty_emb, apply_pos, mask_flag):
        seq_len, batch_size = query.size(1), query.size(0)
        # 上三角和对角为1，其余为0的矩阵
        upper_triangle_ones = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask_flag).astype('uint8')
        src_mask = (torch.from_numpy(upper_triangle_ones) == 0).to(self.params["device"])
        if not mask_flag:
            # 只看过去
            query2 = self.masked_attn_head(query, key, value, src_mask, True, question_difficulty_emb)
        else:
            # 看当前和过去
            query2 = self.masked_attn_head(query, key, value, src_mask, False, question_difficulty_emb)

        # 残差
        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query
    

class TransformerLayer4SparseKT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        model_config = self.params["models_config"]["SparseKT"]
        dim_model = model_config["dim_model"]
        dim_ff = model_config["dim_ff"]
        dropout = model_config["dropout"]

        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention4SparseKT(params)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        mask,
        query,
        key,
        values,
        apply_pos=True,
    ):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """
        seq_len, batch_size = query.size(1), query.size(0)
        no_peek_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(no_peek_mask) == 0).to(self.params["device"])
        if mask == 0:  # If 0, zero-padding is needed.
            # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
            query2, _ = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2, _ = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(
                self.dropout(self.activation(self.linear1(query)))  # FFN
            )
            query = query + self.dropout2(query2)  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query, _
    

class TransformerLayer4CLKT(nn.Module):
    def __init__(self, params):
        super(TransformerLayer4CLKT, self).__init__()
        self.params = params

        model_config = params["models_config"]["CLKT"]
        dim_model = model_config["dim_model"]
        dim_ff = model_config["dim_ff"]
        dropout = model_config["dropout"]

        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention4CLKT(params)

        # Two layer norm and two dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)

        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, mask, query, key, values, apply_pos=True):
        # mask: 0 means that it can peek (留意) only past values. 1 means that block can peek current and past values

        batch_size, seq_len = query.size(0), query.size(1)
        # 从输入矩阵中抽取上三角矩阵，k表示从第几条对角线开始
        upper_tri_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(upper_tri_mask) == 0).to(self.params["device"])
        bert_mask = torch.ones_like(src_mask).bool()

        if mask == 0:
            # 单向的attention，只看过去
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        elif mask == 1:
            # 单向的attention，包括当前
            query2, attn = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)
        else:
            # 双向的attention
            query2, attn = self.masked_attn_head(query, key, values, mask=bert_mask, zero_pad=False)

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)

        return query, attn
