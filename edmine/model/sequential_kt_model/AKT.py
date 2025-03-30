import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.Transformer import TransformerLayer4AKT
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import binary_cross_entropy


class AKT(nn.Module, DLSequentialKTModel):
    model_name = "AKT"

    def __init__(self, params, objects):
        super(AKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["AKT"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.encoder_layer = Architecture(params)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def base_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        separate_qa = self.params["models_config"]["AKT"]["separate_qa"]
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
        separate_qa = self.params["models_config"]["AKT"]["separate_qa"]
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
        loss = predict_loss + rasch_loss * self.params["loss_config"]["rasch loss"]
        
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
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }
    
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, target_question)
        concept_variation_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, target_question)
        question_difficulty_emb = self.embed_layer.get_emb("question_diff", target_question)
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        latent = self.get_latent(batch)
        target_latent = latent[:, target_index-1]
        num_question = target_question.shape[1]
        batch_size = batch["correctness_seq"].shape[0]
        target_latent_extend = target_latent.repeat_interleave(num_question, dim=0).view(batch_size, num_question, -1)
        predict_layer_input = torch.cat((target_latent_extend, question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        return predict_score

    def get_knowledge_state(self, batch):
        pass

class Architecture(nn.Module):
    def __init__(self, params):
        super(Architecture, self).__init__()
        self.params = params
        num_block = self.params["models_config"]["AKT"]["num_block"]

        # question encoder的偶数层（从0开始）用于对question做self attention，奇数层用于对question和interaction做cross attention
        self.question_encoder = nn.ModuleList([TransformerLayer4AKT(params) for _ in range(num_block * 2)])
        self.knowledge_encoder = nn.ModuleList([TransformerLayer4AKT(params) for _ in range(num_block)])

    def get_latent(self, batch):
        y = batch["interaction_emb"]
        for block in self.knowledge_encoder:
            y = block(y, y, y, batch["question_difficulty_emb"], apply_pos=False, mask_flag=True)
        return y

    def forward(self, batch, is_core=False):
        x = batch["question_emb"]
        y = batch["interaction_emb"]
        question_difficulty_emb = batch["question_difficulty_emb"]

        for block in self.knowledge_encoder:
            # 对0～t-1时刻前的qa信息进行编码, \hat{y_t}
            y = block(y, y, y, question_difficulty_emb, apply_pos=True, mask_flag=True)

        flag_first = True
        for block in self.question_encoder:
            if flag_first:
                # peek current question
                # False: 没有FFN, 第一层只有self attention, \hat{x_t}
                x = block(x, x, x, question_difficulty_emb, apply_pos=False, mask_flag=True)
                flag_first = False
            else:
                # don't peek current response
                # True: +FFN+残差+layer norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                x = block(x, x, y, question_difficulty_emb, apply_pos=True, mask_flag=False)
                flag_first = True

        if is_core:
            return x, y
        else:
            return x