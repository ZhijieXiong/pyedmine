import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class DKVMN(nn.Module, DLSequentialKTModel):
    model_name = "DKVMN"

    def __init__(self, params, objects):
        super(DKVMN, self).__init__()
        self.params = params
        self.objects = objects

        model_config = params["models_config"]["DKVMN"]
        dim_key = model_config["embed_config"]["key"]["dim_item"]
        dim_value = model_config["dim_value"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.Mk = nn.Parameter(torch.Tensor(dim_value, dim_key))
        self.Mv0 = nn.Parameter(torch.Tensor(dim_value, dim_key))
        nn.init.kaiming_normal_(self.Mk)
        nn.init.kaiming_normal_(self.Mv0)
        self.f_layer = nn.Linear(dim_key * 2, dim_key)
        self.e_layer = nn.Linear(dim_key, dim_key)
        self.a_layer = nn.Linear(dim_key, dim_key)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        correctness_seq = batch["correctness_seq"]
        batch_size = correctness_seq.shape[0]

        k = self.embed_layer.get_emb_fused1(
            "key", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        interaction_seq = num_concept * batch["correctness_seq"].unsqueeze(-1)
        v = self.embed_layer.get_emb_fused1(
            "value", q2c_transfer_table, q2c_mask_table, batch["question_seq"], other_item_index=interaction_seq)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat([(w.unsqueeze(-1) * Mv[:, :-1]).sum(-2), k], dim=-1)
            )
        )

        predict_score_batch = self.predict_layer(f).squeeze(-1)

        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)[:, seq_start-1:]
        predict_score = torch.masked_select(predict_score_batch, mask_seq[:, seq_start-1:])
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        # todo: 
        latent = self.get_latent(batch)
        target_latent = latent[:, target_index-1]

        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, target_question)

        num_question = target_question.shape[1]
        batch_size = batch["correctness_seq"].shape[0]
        target_latent_extend = target_latent.repeat_interleave(num_question, dim=0).view(batch_size, num_question, -1)
        predict_layer_input = torch.cat((target_latent_extend, concept_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_score_at_target_time(self, batch, target_index):
        # todo: 
        predict_score_batch = self.forward(batch)
        return predict_score_batch[:, target_index-1]

    def get_knowledge_state(self, batch):
        # todo: 
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
