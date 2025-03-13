import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer


class DIMKT(nn.Module, DLSequentialKTModel):
    model_name = "DIMKT"

    def __init__(self, params, objects):
        super(DIMKT, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["DIMKT"]
        dim_emb = model_config["embed_config"]["question"]["dim_item"]
        dropout = model_config["dropout"]
        
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.generate_x_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.SDF_MLP1 = nn.Linear(dim_emb, dim_emb)
        self.SDF_MLP2 = nn.Linear(dim_emb, dim_emb)
        self.PKA_MLP1 = nn.Linear(2 * dim_emb, dim_emb)
        self.PKA_MLP2 = nn.Linear(2 * dim_emb, dim_emb)
        self.knowledge_indicator_MLP = nn.Linear(4 * dim_emb, dim_emb)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        q2c_diff_table = self.objects["dimkt"]["q2c_diff_table"]
        dim_emb = self.params["models_config"]["DIMKT"]["embed_config"]["question"]["dim_item"]
        batch_size, seq_len = batch["question_seq"].shape[0], batch["question_seq"].shape[1]

        question_emb = self.embed_layer.get_emb("question", batch["question_seq"])
        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        question_diff_emb = self.embed_layer.get_emb("question_diff", batch["question_diff_seq"])
        concept_diff_emb = self.embed_layer.get_emb_fused2("concept_diff", q2c_transfer_table, q2c_mask_table, q2c_diff_table, batch["question_seq"])
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])

        latent = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        h_pre = nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(self.params["device"])
        y = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len-1):
            input_x = torch.cat((
                question_emb[:, t],
                concept_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            x = self.generate_x_MLP(input_x)
            input_sdf = x - h_pre
            sdf = torch.sigmoid(self.SDF_MLP1(input_sdf)) * self.dropout_layer(torch.tanh(self.SDF_MLP2(input_sdf)))
            input_pka = torch.cat((sdf, correctness_emb[:, t]), dim=1)
            pka = torch.sigmoid(self.PKA_MLP1(input_pka)) * torch.tanh(self.PKA_MLP2(input_pka))
            input_KI = torch.cat((
                h_pre,
                correctness_emb[:, t],
                question_diff_emb[:, t],
                concept_diff_emb[:, t]
            ), dim=1)
            Gamma_ksu = torch.sigmoid(self.knowledge_indicator_MLP(input_KI))
            h = Gamma_ksu * h_pre + (1 - Gamma_ksu) * pka
            input_x_next = torch.cat((
                question_emb[:, t+1],
                concept_emb[:, t+1],
                question_diff_emb[:, t+1],
                concept_diff_emb[:, t+1]
            ), dim=1)
            x_next = self.generate_x_MLP(input_x_next)
            y[:, t] = torch.sigmoid(torch.sum(x_next * h, dim=-1))
            latent[:, t + 1, :] = h
            h_pre = h

        return y

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)[:, seq_start-2:-1]
        predict_score = torch.masked_select(predict_score_batch, mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        latent = self.get_latent(batch)
        target_latent = latent[:, target_index-1]

        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        target_question_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, target_question)

        num_question = target_question.shape[1]
        batch_size = batch["correctness_seq"].shape[0]
        target_latent_extend = target_latent.repeat_interleave(num_question, dim=0).view(batch_size, num_question, -1)
        predict_layer_input = torch.cat((target_latent_extend, target_question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_score_at_target_time(self, batch, target_index):
        predict_score_batch = self.forward(batch)
        return predict_score_batch[:, target_index-1]

    def get_knowledge_state(self, batch):
        num_concept = self.params["models_config"]["DKT"]["embed_config"]["concept"]["num_item"]

        self.encoder_layer.flatten_parameters()
        batch_size = batch["correctness_seq"].shape[0]
        first_index = torch.arange(batch_size).long().to(self.params["device"])
        all_concept_id = torch.arange(num_concept).long().to(self.params["device"])
        all_concept_emb = self.embed_layer.get_emb("concept", all_concept_id)

        latent = self.get_latent(batch)
        last_latent = latent[first_index, batch["seq_len"] - 2]
        last_latent_expanded = last_latent.repeat_interleave(num_concept, dim=0).view(batch_size, num_concept, -1)
        all_concept_emb_expanded = all_concept_emb.expand(batch_size, -1, -1)
        predict_layer_input = torch.cat([last_latent_expanded, all_concept_emb_expanded], dim=-1)

        return self.predict_layer(predict_layer_input).squeeze(dim=-1)

