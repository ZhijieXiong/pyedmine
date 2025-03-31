import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer


class CausalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=2 * dim,  # 输出双倍通道用于GLU分割
            kernel_size=kernel_size,
            padding=0  # 手动进行左padding保持因果性
        )
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):       
        # 转换为Conv1d需要的形状 (batch_size, dim, seq_len)
        x_conv = x.permute(0, 2, 1)
        
        # 因果卷积的左padding（kernel_size-1个零）
        x_padded = F.pad(x_conv, (self.kernel_size - 1, 0))
        
        # 一维卷积
        conv_out = self.conv(x_padded)  # (batch_size, 2*dim, seq_len)
        
        # 恢复形状并应用GLU
        conv_out = conv_out.permute(0, 2, 1)  # (batch_size, seq_len, 2*dim)
        glu_out = F.glu(conv_out, dim=-1)     # (batch_size, seq_len, dim)
        
        # 残差连接
        return glu_out + self.dropout(x)


class CausalConvNet(nn.Module):
    def __init__(self, dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalConvBlock(dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        for layer in self.layers:
            x = layer(x)
        return x


class CKT(nn.Module, DLSequentialKTModel):
    model_name = "CKT"

    def __init__(self, params, objects):
        super(CKT, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["CKT"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        dim_emb = model_config["embed_config"]["question"]["dim_item"]
        num_layer = model_config["num_layer"]
        kernel_size = model_config["kernel_size"]
        dropout = model_config["dropout"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.W1 = nn.Linear(4 * dim_emb + num_concept, dim_emb)
        self.W2 = nn.Linear(4 * dim_emb + num_concept, dim_emb)
        self.encoder_layer = CausalConvNet(dim_emb, num_layer, kernel_size, dropout)
        
    def forward(self, batch):
        question_emb = self.embed_layer.get_emb("question", batch["question_seq"])
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        
        LIS = torch.cat((question_emb, correctness_emb), dim=-1)
        CPC = batch["cqc_seq"]
        
        # Mask Softmax
        _, seq_len, _ = question_emb.shape
    
        scores = torch.matmul(question_emb, question_emb.transpose(-2, -1))  # (bs, seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # 上三角为True
        mask = mask.unsqueeze(0).to(self.params["device"])
        scores = scores.masked_fill(mask, float('-inf'))
        W = F.softmax(scores, dim=-1)  # (bs, seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()  # 上三角为True
        mask = mask.unsqueeze(0).to(self.params["device"])
        W = W.masked_fill(mask, 0)
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
        HRP = torch.bmm(W, LIS)
        
        H = torch.cat((LIS, HRP, CPC), dim=-1)
        Q = self.W1(H) * torch.sigmoid(self.W2(H))
        latent = self.encoder_layer(Q)
        predict_score_batch = torch.sigmoid((latent[:, :-1] * question_emb[:, 1:]).sum(dim=2))
        return predict_score_batch
    
    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass

        
        
        