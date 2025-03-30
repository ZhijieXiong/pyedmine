import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer, CosinePositionalEmbedding
from edmine.model.module.Transformer import TransformerLayer4SimpleKT
from edmine.model.module.PredictorLayer import PredictorLayer


class SimpleKT(nn.Module, DLSequentialKTModel):
    model_name = "SimpleKT"

    def __init__(self, params, objects):
        super(SimpleKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["SimpleKT"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.encoder_layer = Architecture(params)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def base_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        separate_qa = self.params["models_config"]["SimpleKT"]["separate_qa"]
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
        # e_{(c_t, r_t)} + mu_{q_t} * f_{(c_t, r_t)}
        interaction_emb = (interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb))

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
        super().__init__()

        self.params = params
        model_config = self.params["models_config"]["SimpleKT"]
        num_block = model_config["num_block"]
        dim_model = model_config["dim_model"]
        seq_len = model_config["seq_len"]

        self.dim_model = dim_model
        self.blocks = nn.ModuleList([TransformerLayer4SimpleKT(params) for _ in range(num_block)])
        self.position_emb = CosinePositionalEmbedding(d_model=self.dim_model, max_len=seq_len)

    def get_latent(self, batch):
        interaction_emb = batch["interaction_emb"]
        emb_position_interaction = self.position_emb(interaction_emb)
        interaction_emb = interaction_emb + emb_position_interaction
        y = interaction_emb

        for block in self.blocks:
            # apply_pos is True: FFN+残差+lay norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，第一题只有question信息，无interaction信息
            y = block(mask=1, query=y, key=y, values=y, apply_pos=True)

        return y

    def forward(self, batch):
        question_emb = batch["question_emb"]
        interaction_emb = batch["interaction_emb"]

        # target shape: (batch_size, seq_len)
        emb_position_question = self.position_emb(question_emb)
        question_emb = question_emb + emb_position_question
        emb_position_interaction = self.position_emb(interaction_emb)
        interaction_emb = interaction_emb + emb_position_interaction

        y = interaction_emb
        x = question_emb

        for block in self.blocks:
            # apply_pos is True: FFN+残差+lay norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，第一题只有question信息，无interaction信息
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x