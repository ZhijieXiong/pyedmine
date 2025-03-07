import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer


class DKT_KG4EX(nn.Module):
    model_name = "DKT_KG4EX"
    model_type = "DLSequentialKTModel"

    def __init__(self, params, objects):
        super(DKT_KG4EX, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["DKT_KG4EX"]
        embed_config = model_config["embed_config"]
        dim_concept = embed_config["concept"]["dim_item"]
        dim_correctness = embed_config["correctness"]["dim_item"]
        dim_emb = dim_concept + dim_correctness
        dim_latent = model_config["dim_latent"]
        rnn_type = model_config["rnn_type"]
        num_rnn_layer = model_config["num_rnn_layer"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_emb, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def get_latent(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        self.encoder_layer.flatten_parameters()
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        interaction_emb = torch.cat((concept_emb[:, :-1], correctness_emb[:, :-1]), dim=2)
        latent, _ = self.encoder_layer(interaction_emb)

        return latent

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        latent = self.get_latent(batch)
        predict_layer_input = torch.cat((latent, concept_emb[:, 1:]), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_bool_seq[:, seq_start-1:])
        # mac M1不支持double
        if self.params["device"] == "mps":
            predict_loss = nn.functional.binary_cross_entropy(predict_score.float(), ground_truth.float())
        else:
            predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        aux_loss = self.get_aux_loss(batch)
        loss = predict_loss + self.params["loss_config"]["aux loss"] * aux_loss

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "aux loss": {
                    "value": aux_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_aux_loss(self, batch):
        """防止模型全输出1（原论文代码提供的案例中单个学生最后时刻的pkc之和不为1，所以不是概率分布，不采用多分类损失，并且论文中也不是用的多分类损失）"""
        num_concept = self.params["models_config"]["DKT_KG4EX"]["embed_config"]["concept"]["num_item"]
        dim_latent = self.params["models_config"]["DKT_KG4EX"]["dim_latent"]

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        batch_size = batch["correctness_seq"].shape[0]
        seq_len = batch["correctness_seq"].shape[1]
        all_concept_id = torch.arange(num_concept).long().to(self.params["device"])
        all_concept_emb = self.embed_layer.get_emb("concept", all_concept_id)

        latent = self.get_latent(batch)
        latent_expanded = latent.repeat_interleave(num_concept, dim=1).view(batch_size, -1, num_concept, dim_latent)
        all_concept_emb_expanded = all_concept_emb.expand(batch_size, seq_len - 1, -1, -1)
        predict_layer_input = torch.cat([latent_expanded, all_concept_emb_expanded], dim=-1)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        mask_expanded = mask_bool_seq[:, 1:].unsqueeze(-1).repeat_interleave(num_concept, dim=-1)
        predict_score = predict_score_batch[mask_expanded.bool()]

        return torch.mean(predict_score)

    def get_knowledge_state(self, batch):
        num_concept = self.params["models_config"]["DKT_KG4EX"]["embed_config"]["concept"]["num_item"]

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
