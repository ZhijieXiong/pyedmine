import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import binary_cross_entropy
from edmine.model.module.Transformer import TransformerLayer4RouterKT

class RouterKT(nn.Module, DLSequentialKTModel):
    model_name = "RouterKT"

    def __init__(self, params, objects):
        super(RouterKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["RouterKT"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.encoder_layer = RouterKTArchitecture(params)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])
        
    def base_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        separate_qa = self.params["models_config"]["RouterKT"]["separate_qa"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]

        # c_ct
        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        if separate_qa:
            interaction_seq = num_concept * batch["correctness_seq"].unsqueeze(-1)
            interaction_emb = self.embed_layer.get_emb_fused1(
                "interaction", q2c_transfer_table, q2c_mask_table, batch["question_seq"], other_item_index=interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_layer.get_emb("interaction", batch["correctness_seq"]) + concept_emb

        return concept_emb, interaction_emb

    def get_latent(self, batch):
        separate_qa = self.params["models_config"]["RouterKT"]["separate_qa"]
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        # mu_{q_t}
        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        # f_{(c_t, r_t)}中的r_t
        interaction_variation_emb = self.embed_layer.get_emb("interaction_var", batch["correctness_seq"])
        if separate_qa:
            # uq * f_(ct,rt) + e_(ct,rt)
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)

        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }

        latent = self.encoder_layer(encoder_input)
        return latent

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        concept_emb, _ = self.base_emb(batch)
        concept_variation_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        latent = self.get_latent(batch)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation
        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch, seq_start)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])

        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        rasch_loss = (question_difficulty_emb[mask_seq] ** 2.).sum()

        # Get balance loss from MoH attention layers
        balance_loss = self.encoder_layer.get_balance_loss()

        # Combine losses
        model_config = self.params["models_config"]["RouterKT"]
        loss = predict_loss + \
               rasch_loss * self.params["loss_config"]["rasch loss"] + \
               balance_loss * model_config["balance_loss_weight"]

        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "rasch loss": {
                    "value": rasch_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "balance loss": {
                    "value": balance_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_knowledge_state(self, batch):
        pass


class RouterKTArchitecture(nn.Module):
    def __init__(self, params):
        super(RouterKTArchitecture, self).__init__()
        self.params = params
        model_config = self.params["models_config"]["RouterKT"]
        num_block = model_config["num_block"]

        # Create transformer blocks with MoH attention
        self.question_encoder = nn.ModuleList([
            TransformerLayer4RouterKT(params) for _ in range(num_block * 2)
        ])
        self.knowledge_encoder = nn.ModuleList([
            TransformerLayer4RouterKT(params) for _ in range(num_block)
        ])

    def get_balance_loss(self):
        """Get balance loss from all MoH attention layers."""
        balance_loss = 0.0
        for block in self.knowledge_encoder:
            balance_loss += block.attn.get_balance_loss()
        for block in self.question_encoder:
            balance_loss += block.attn.get_balance_loss()
        return balance_loss

    def forward(self, batch):
        x = batch["question_emb"]
        y = batch["interaction_emb"]
        # question_difficulty_emb = batch["question_difficulty_emb"]
        diff = batch["question_difficulty_emb"]
        response = None

        # 始终只使用 Question 信息做路由
        # Knowledge encoder
        for block in self.knowledge_encoder:
            # Process interaction embeddings
            y = block(y, y, y, mask_flag=True, diff=diff, response=response, apply_pos=True, q4router=x)

        # Question encoder with alternating self-attention and cross-attention
        flag_first = True
        for block in self.question_encoder:
            if flag_first:
                # Self-attention on question embeddings
                x = block(x, x, x, mask_flag=True, diff=diff, response=response, apply_pos=False, q4router=x)
                flag_first = False
            else:
                # Cross-attention between question and interaction
                x = block(x, x, y, mask_flag=False, diff=diff, response=response, apply_pos=True, q4router=x)
                flag_first = True

        return x


