import torch
from torch import nn


from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer, CosinePositionalEmbedding
from edmine.model.module.Transformer import TransformerLayer4SparseKT
from edmine.model.module.PredictorLayer import PredictorLayer


class SparseKT(nn.Module, DLSequentialKTModel):
    model_name = "SparseKT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["SparseKT"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.encoder_layer = Architecture(params)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def base_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        separate_qa = self.params["models_config"]["SparseKT"]["separate_qa"]
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
        separate_qa = self.params["models_config"]["SparseKT"]["separate_qa"]
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

        latent = self.encoder_layer(question_emb, interaction_emb)
        return latent
    
    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, _ = self.base_emb(batch)
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        # mu_{q_t}
        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        latent = self.get_latent(batch)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

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

        model_config = self.params["models_config"]["SparseKT"]
        dim_model = model_config["dim_model"]
        num_block = model_config["num_block"]
        seq_len = model_config["seq_len"]

        self.blocks_2 = nn.ModuleList([TransformerLayer4SparseKT(params) for _ in range(num_block)])
        self.position_emb = CosinePositionalEmbedding(d_model=dim_model, max_len=seq_len)

    def forward(
        self,
        question_emb,
        interaction_emb,
    ):
        question_po_emb = self.position_emb(question_emb)
        question_emb = question_emb + question_po_emb
        interaction_pos_emb = self.position_emb(interaction_emb)
        interaction_emb = interaction_emb + interaction_pos_emb

        qa_pos_embed = interaction_emb
        q_pos_embed = question_emb

        y = qa_pos_embed
        x = q_pos_embed

        for block in self.blocks_2:
            # True: +FFN+残差+layer norm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retriever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
            x, attn_weights = block(
                mask=0,
                query=x,
                key=x,
                values=y,
                apply_pos=True,
            )
        return x