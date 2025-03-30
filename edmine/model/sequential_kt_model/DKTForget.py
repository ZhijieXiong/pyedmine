import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class DKTForget(nn.Module, DLSequentialKTModel):
    model_name = "DKTForget"

    def __init__(self, params, objects):
        super(DKTForget, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["DKTForget"]
        embed_config = model_config["embed_config"]
        dim_emb = embed_config["concept"]["dim_item"]
        dim_latent = model_config["dim_latent"]
        rnn_type = model_config["rnn_type"]
        num_rnn_layer = model_config["num_rnn_layer"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.C1 = nn.Linear(dim_emb * 2, dim_emb)
        self.C2 = nn.Linear(dim_emb + dim_latent, dim_emb)
        self.C3 = nn.Linear(dim_emb * 3, dim_emb)
        dim_encoder_in = dim_emb * 4
        if rnn_type == "rnn":
            self.encoder_layer = nn.RNN(dim_encoder_in, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.encoder_layer = nn.LSTM(dim_encoder_in, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.encoder_layer = nn.GRU(dim_encoder_in, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def get_latent(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        self.encoder_layer.flatten_parameters()
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        interaction_emb = self.C1(torch.cat((concept_emb, correctness_emb), dim=-1))[:, :-1]
        
        repeat_it_emb = self.embed_layer.get_emb("repeat_interval_time", batch["repeat_interval_time_seq"])
        it_emb = self.embed_layer.get_emb("interval_time", batch["interval_time_seq"])
        num_repeat_emb = self.embed_layer.get_emb("num_repeat", batch["num_repeat_seq"])
        other_emb = torch.cat((repeat_it_emb, it_emb, num_repeat_emb), dim=-1)[:, :-1]
        
        encoder_in_emb = torch.cat((interaction_emb * self.C3(other_emb), other_emb), dim=-1)
        latent, _ = self.encoder_layer(encoder_in_emb)

        return latent

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])[:, 1:]
        
        repeat_it_emb = self.embed_layer.get_emb("repeat_interval_time", batch["repeat_interval_time_seq"])
        it_emb = self.embed_layer.get_emb("interval_time", batch["interval_time_seq"])
        num_repeat_emb = self.embed_layer.get_emb("num_repeat", batch["num_repeat_seq"])
        other_emb = torch.cat((repeat_it_emb, it_emb, num_repeat_emb), dim=-1)[:, 1:]
        
        latent = self.get_latent(batch)
        next_emb = self.C2(torch.cat((latent, concept_emb), dim=-1))
        
        predict_in_emb = torch.cat((next_emb * self.C3(other_emb), other_emb), dim=-1)
        predict_score_batch = self.predict_layer(predict_in_emb).squeeze(dim=-1)

        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass
