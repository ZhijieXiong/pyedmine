import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer


class QDCKT(nn.Module, DLSequentialKTModel):
    model_name = "QDCKT"

    def __init__(self, params, objects):
        super(QDCKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["QDCKT"]
        dim_concept = model_config["embed_config"]["concept"]["dim_item"]
        dim_correctness = model_config["embed_config"]["correctness"]["dim_item"]
        dim_que_diff = model_config["embed_config"]["question_diff"]["dim_item"]
        dim_latent = model_config["dim_latent"]
        rnn_type = model_config["rnn_type"]
        num_rnn_layer = model_config["num_rnn_layer"]
        dropout = model_config["dropout"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.dropout_layer = nn.Dropout(dropout)
        self.W = nn.Linear(dim_concept + dim_que_diff, dim_correctness)
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_correctness, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_correctness, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_correctness, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        q2diff_transfer_table = self.objects["dataset"]["q2diff_transfer_table"]
        q2diff_weight_table = self.objects["dataset"]["q2diff_weight_table"]

        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        embed_question_diff = self.embed_layer.__getattr__("question_diff")
        question_diff_emb = embed_question_diff(q2diff_transfer_table[batch["question_seq"]])
        weight = q2diff_weight_table[batch["question_seq"]]
        question_diff_emb = (question_diff_emb * weight.unsqueeze(-1)).sum(-2)
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        concept_que_diff_emb = self.W(torch.cat((concept_emb, question_diff_emb), dim=-1))

        interaction_emb = self.dropout_layer(concept_que_diff_emb[:, :-1]) + correctness_emb[:, :-1]
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        predict_layer_input = torch.cat((latent, self.dropout_layer(concept_que_diff_emb[:, 1:])), dim=-1)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score_batch

