import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.Transformer import TransformerLayer4DisKT
from edmine.model.module.EmbedLayer import EmbedLayer, CosinePositionalEmbedding
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.module.attention import contradictory_attention4dis_kt
from edmine.model.loss import binary_cross_entropy
from edmine.model.registry import register_model

MODEL_NAME = "DisKT"


@register_model(MODEL_NAME)
class DisKT(nn.Module, DLSequentialKTModel):
    model_name = MODEL_NAME

    def __init__(self, params, objects):
        super(DisKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"][MODEL_NAME]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.model = Architecture(params)
        self.ffn = FeedForward(params)
        self.dual_attention = DualAttention(params)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def base_emb(self, concept_seq, target):
        model_config = self.params["models_config"][MODEL_NAME]
        num_concept = model_config["num_concept"]
        separate_qa = model_config["separate_qa"]

        concept_emb = self.embed_layer.get_emb("concept", concept_seq)
        if separate_qa:
            qa_data = concept_seq + num_concept * target
            interaction_emb = self.embed_layer.get_emb("interaction", qa_data)
        else:
            interaction_emb = self.embed_layer.get_emb("interaction", target) + concept_emb

        return concept_emb, interaction_emb

    def rasch_emb(self, concept_seq, question_seq, target):
        concept_emb, interaction_emb = self.base_emb(concept_seq, target)
        concept_var_emb = self.embed_layer.get_emb("concept_var", concept_seq)
        question_diff_emb = self.embed_layer.get_emb("question_diff", question_seq)
        question_emb = concept_emb + question_diff_emb * concept_var_emb

        interaction_var_emb = self.embed_layer.get_emb("interaction_var", target)
        interaction_emb = interaction_emb + question_diff_emb * interaction_var_emb

        return question_emb, interaction_emb, question_diff_emb

    def forward(self, batch):
        question_seq = batch['question_seq']
        concept_seq = batch['concept_seq']
        correctness_seq = batch['correctness_seq']
        counter_mask_seq = batch['counter_mask_seq']
        masked_r = correctness_seq * (correctness_seq > -1).long()

        pos_question_emb, pos_interaction_emb, _ = self.rasch_emb(
            masked_r * concept_seq, masked_r * question_seq, 2 - masked_r)
        neg_question_emb, neg_interaction_emb, _ = self.rasch_emb(
            (1 - masked_r) * concept_seq, (1 - masked_r) * question_seq, 2 * masked_r)
        question_emb, interaction_emb, question_diff_emb = self.rasch_emb(concept_seq, question_seq, masked_r)

        y1, y2, y = pos_interaction_emb, neg_interaction_emb, interaction_emb
        x = question_emb
        x = self.model(x, y)
        y1, y2 = self.ffn(y1), self.ffn(y2)
        y1, y2 = self.dual_attention(x, x, y1, y2, counter_mask_seq)
        x = x - (y1 + y2)
        x = x - question_diff_emb
        x = torch.cat([x, question_emb], dim=-1)
        x = torch.cat([x, y1 - y2], dim=-1)
        output = self.predict_layer(x).squeeze(-1)
        predict_score_batch = torch.sigmoid(output)

        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        question_seq = batch['question_seq_new']
        concept_seq = batch['concept_seq_new']
        correctness_seq = batch['correctness_seq_new']
        counter_mask_seq = batch['counter_mask_seq']
        masked_r = correctness_seq * (correctness_seq > -1).long()

        _, pos_interaction_emb, _ = self.rasch_emb(
            masked_r * concept_seq, masked_r * question_seq, 2 - masked_r)
        _, neg_interaction_emb, _ = self.rasch_emb(
            (1 - masked_r) * concept_seq, (1 - masked_r) * question_seq, 2 * masked_r)
        question_emb, interaction_emb, question_diff_emb = self.rasch_emb(concept_seq, question_seq, masked_r)

        y1, y2, y = pos_interaction_emb, neg_interaction_emb, interaction_emb
        x = question_emb
        x = self.model(x, y)
        y1, y2 = self.ffn(y1), self.ffn(y2)
        y1, y2 = self.dual_attention(x, x, y1, y2, counter_mask_seq)
        x = x - (y1 + y2)
        x = x - question_diff_emb
        x = torch.cat([x, question_emb], dim=-1)
        x = torch.cat([x, y1 - y2], dim=-1)
        output = self.predict_layer(x).squeeze(-1)
        predict_score_batch = torch.sigmoid(output)

        # DisKT是在前面填充的padding，对其进行移位
        # 计算每行需要移动的位数
        max_seq_len = question_seq.shape[1]
        shift = max_seq_len - batch["seq_len"]
        idx = (torch.arange(max_seq_len).unsqueeze(0).to(self.params["device"]) + shift.unsqueeze(1)) % max_seq_len
        # gather 按索引取值
        predict_score_batch = torch.gather(predict_score_batch, 1, idx)[:, 1:]

        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation

        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        question_seq = batch['question_seq_new']
        concept_seq = batch['concept_seq_new']
        correctness_seq = batch['correctness_seq_new']
        counter_mask_seq = batch['counter_mask_seq']
        masked_r = correctness_seq * (correctness_seq > -1).long()

        _, pos_interaction_emb, _ = self.rasch_emb(
            masked_r * concept_seq, masked_r * question_seq, 2 - masked_r)
        _, neg_interaction_emb, _ = self.rasch_emb(
            (1 - masked_r) * concept_seq, (1 - masked_r) * question_seq, 2 * masked_r)
        question_emb, interaction_emb, question_diff_emb = self.rasch_emb(concept_seq, question_seq, masked_r)

        y1, y2, y = pos_interaction_emb, neg_interaction_emb, interaction_emb
        x = question_emb
        distance = F.pairwise_distance(y1.view(y1.size(0), -1), y2.view(y2.size(0), -1))
        reg_loss = torch.mean(distance)
        num_reg_sample = y1.size(0)

        x = self.model(x, y)
        y1, y2 = self.ffn(y1), self.ffn(y2)
        y1, y2 = self.dual_attention(x, x, y1, y2, counter_mask_seq)
        x = x - (y1 + y2)
        x = x - question_diff_emb
        x = torch.cat([x, question_emb], dim=-1)
        x = torch.cat([x, y1 - y2], dim=-1)
        output = self.predict_layer(x).squeeze(-1)
        predict_score_batch = torch.sigmoid(output)

        # DisKT是在前面填充的padding，对其进行移位
        # 计算每行需要移动的位数
        max_seq_len = question_seq.shape[1]
        shift = max_seq_len - batch["seq_len"]
        idx = (torch.arange(max_seq_len).unsqueeze(0).to(self.params["device"]) + shift.unsqueeze(1)) % max_seq_len
        # gather 按索引取值
        predict_score_batch = torch.gather(predict_score_batch, 1, idx)[:, 1:]

        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start - 2:], mask_seq[:, seq_start - 1:])
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start - 1:], mask_seq[:, seq_start - 1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])

        loss = predict_loss + reg_loss * self.params["loss_config"]["reg loss"]

        num_sample = torch.sum(batch["mask_seq"][:, seq_start - 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "reg loss": {
                    "value": reg_loss.detach().cpu().item() * num_reg_sample,
                    "num_sample": num_reg_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_knowledge_state(self, batch):
        pass


class Architecture(nn.Module):
    def __init__(self, params):
        super(Architecture, self).__init__()

        self.params = params
        model_config = self.params["models_config"][MODEL_NAME]
        dim_model = model_config["dim_model"]
        num_block = model_config["num_block"]
        seq_len = model_config["seq_len"]

        self.embedding_size = dim_model
        self.blocks_2 = nn.ModuleList([TransformerLayer4DisKT(params) for _ in range(num_block)])
        self.position_emb = CosinePositionalEmbedding(self.embedding_size, seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        q_position_emb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_position_emb
        qa_position_emb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_position_emb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        for block in self.blocks_2:
            # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class DualAttention(nn.Module):
    def __init__(self, params):
        super(DualAttention, self).__init__()

        self.params = params
        model_config = self.params["models_config"][MODEL_NAME]
        dim_model = model_config["dim_model"]
        num_head = model_config["num_head"]
        dropout = model_config["dropout"]

        assert dim_model % num_head == 0
        self.n_heads = num_head
        self.d_model = dim_model
        self.n_feature = dim_model // num_head
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v1, v2, counter_attention_mask):
        batch_size = q.size(0)
        src_mask = create_mask(q, 0)
        q = q.view(batch_size, -1, self.n_heads, self.n_feature)
        k = k.view(batch_size, -1, self.n_heads, self.n_feature)
        v1 = v1.view(batch_size, -1, self.n_heads, self.n_feature)
        v2 = v2.view(batch_size, -1, self.n_heads, self.n_feature)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)
        output_v1, output_v2, attn_weight = contradictory_attention4dis_kt(
            q, k, v1, v2, src_mask, self.dropout, counter_attention_mask)
        output_v1 = output_v1.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_v2 = output_v2.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return output_v1, output_v2


class FeedForward(nn.Module):
    def __init__(self, params):
        super(FeedForward, self).__init__()

        self.params = params
        model_config = self.params["models_config"][MODEL_NAME]
        dim_model = model_config["dim_model"]
        dropout = model_config["dropout"]
        inner_size = dim_model * 2

        self.w_1 = nn.Linear(dim_model, inner_size)
        self.w_2 = nn.Linear(inner_size, dim_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(dim_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


def create_mask(x, mask):
    seq_len = x.size(1)
    device = x.device
    no_peek_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype('uint8')
    src_mask = (torch.from_numpy(no_peek_mask) == 0).to(device)
    return src_mask