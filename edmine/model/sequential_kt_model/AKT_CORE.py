import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.sequential_kt_model.AKT import Architecture
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.registry import register_model

MODEL_NAME = "AKT"


@register_model("AKT_CORE")
class AKT_CORE(nn.Module, DLSequentialKTModel):
    model_name = "AKT_CORE"

    def __init__(self, params, objects):
        super(AKT_CORE, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"][MODEL_NAME]
        dim_model = model_config["dim_model"]
        dim_ff = model_config["dim_ff"]
        dropout = model_config["dropout"]
        
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.encoder_layer = Architecture(params)
        self.predict_layer = nn.Sequential(
            nn.Linear(dim_model + dim_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
        self.question_net = nn.Sequential(
            nn.Linear(dim_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        self.user_net = nn.Linear(dim_model, 2)
        self.constant = nn.Parameter(torch.tensor(0.0))

    def base_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        separate_qa = self.params["models_config"][MODEL_NAME]["separate_qa"]
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
    
    def forward(self, batch):
        separate_qa = self.params["models_config"][MODEL_NAME]["separate_qa"]
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

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
        d_output, s_output = self.encoder_layer(encoder_input, is_core=True)
        
        concat_q = torch.cat([d_output, question_emb], dim=-1)
        output = self.predict_layer(concat_q)
        q_logit = self.question_net(question_emb.detach())[:, 1:]
        s_logit = self.user_net(s_output.detach())[:, :-1]
        logits = output[:, 1:]
        z_qks = self.fusion(logits, q_logit, s_logit, Q_fact=True, K_fact=True, S_fact=True)
        z_q = self.fusion(logits, q_logit, s_logit, Q_fact=True, K_fact=False, S_fact=False)
        logit_core = z_qks - z_q

        z_nde = self.fusion(logits.clone().detach(), q_logit.clone().detach(), s_logit.clone().detach(),
                            Q_fact=True, K_fact=False, S_fact=False)
        # NDE = z_Q - z
        mask_bool_seq_ = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        z_nde_pred = torch.masked_select(z_nde, mask_bool_seq_).view(-1, 2)
        q_pred = torch.masked_select(q_logit, mask_bool_seq_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_qks, mask_bool_seq_).view(-1, 2)

        return z_nde_pred, q_pred, z_qks_pred, logit_core

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        _, _, _, logit_core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_core, dim=-1)[:, :, 1]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start - 2:], mask_bool_seq[:, seq_start - 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        z_nde_pred, q_pred, z_qks_pred, logit_core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_core, dim=-1)[:, :, 1]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start - 2:], mask_bool_seq[:, seq_start - 1:])
        ground_truth = torch.masked_select(batch["correctness_seq"][:, 1:], mask_bool_seq[:, 1:])

        predict_loss = torch.nn.functional.cross_entropy(z_qks_pred, ground_truth) + \
                       torch.nn.functional.cross_entropy(q_pred, ground_truth)

        p_te = torch.softmax(z_qks_pred, dim=-1).clone().detach()
        KL_loss = - p_te * torch.softmax(z_nde_pred, dim=-1).log()
        KL_loss = KL_loss.sum(1).mean()

        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        rasch_loss = (question_difficulty_emb ** 2.).sum()

        loss = predict_loss + KL_loss + rasch_loss * self.params["loss_config"]["rasch loss"]

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "KL loss": {
                    "value": KL_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "rasch loss": {
                    "value": rasch_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        
    def fusion(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"][MODEL_NAME]["fusion_mode"]
        predict_K, predict_Q, predict_S = self.transform(predict_K, predict_Q, predict_S, Q_fact, K_fact, S_fact)

        if fusion_mode == 'rubin':
            # 鲁宾因果模型（潜在因果框架）
            z = predict_K * torch.sigmoid(predict_Q)

        elif fusion_mode == 'hm':
            #
            z = predict_K * predict_S * predict_Q
            z = torch.log(z + 1e-12) - torch.log1p(z)

        elif fusion_mode == 'sum':
            z = predict_K + predict_Q + predict_S
            z = torch.log(torch.sigmoid(z) + 1e-12)

        else:
            raise NotImplementedError()

        return z

    def transform(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"][MODEL_NAME]["fusion_mode"]

        if not K_fact:
            predict_K = self.constant * torch.ones_like(predict_K).to(self.params["device"])

        if not Q_fact:
            predict_Q = self.constant * torch.ones_like(predict_Q).to(self.params["device"])

        if not S_fact:
            predict_S = self.constant * torch.ones_like(predict_S).to(self.params["device"])

        if fusion_mode == 'hm':
            predict_K = torch.sigmoid(predict_K)
            predict_Q = torch.sigmoid(predict_Q)
            predict_S = torch.sigmoid(predict_S)

        return predict_K, predict_Q, predict_S
    
    def get_knowledge_state(self, batch):
        pass
