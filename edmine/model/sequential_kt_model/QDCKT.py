import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import binary_cross_entropy


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
        
    def get_latent(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        q2diff_transfer_table = self.objects["qdckt"]["q2diff_transfer_table"]
        q2diff_weight_table = self.objects["qdckt"]["q2diff_weight_table"]

        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        embed_question_diff = self.embed_layer.__getattr__("question_diff")
        question_diff_emb = embed_question_diff(q2diff_transfer_table[batch["question_diff_seq"]])
        weight = q2diff_weight_table[batch["question_diff_seq"]]
        question_diff_emb = (question_diff_emb * weight.unsqueeze(-1)).sum(-2)
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        concept_que_diff_emb = self.W(torch.cat((concept_emb, question_diff_emb), dim=-1))

        interaction_emb = self.dropout_layer(concept_que_diff_emb) + correctness_emb
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)
        return latent


    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        q2diff_transfer_table = self.objects["qdckt"]["q2diff_transfer_table"]
        q2diff_weight_table = self.objects["qdckt"]["q2diff_weight_table"]

        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        embed_question_diff = self.embed_layer.__getattr__("question_diff")
        question_diff_emb = embed_question_diff(q2diff_transfer_table[batch["question_diff_seq"]])
        weight = q2diff_weight_table[batch["question_diff_seq"]]
        question_diff_emb = (question_diff_emb * weight.unsqueeze(-1)).sum(-2)
        concept_que_diff_emb = self.W(torch.cat((concept_emb, question_diff_emb), dim=-1))

        latent = self.get_latent(batch)

        predict_layer_input = torch.cat((latent[:, :-1], self.dropout_layer(concept_que_diff_emb[:, 1:])), dim=-1)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        
    def get_predict_loss(self, batch, seq_start=2):
        num_question_diff = self.objects["qdckt"]["num_question_diff"]
        w_qdckt_loss = self.params["loss_config"]["qdckt loss"]
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        q2diff_transfer_table = self.objects["qdckt"]["q2diff_transfer_table"]
        q2diff_weight_table = self.objects["qdckt"]["q2diff_weight_table"]
        mask_seq = torch.ne(batch["mask_seq"], 0)

        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        embed_question_diff = self.embed_layer.__getattr__("question_diff")
        question_diff_emb = embed_question_diff(q2diff_transfer_table[batch["question_diff_seq"]])
        weight = q2diff_weight_table[batch["question_diff_seq"]]
        question_diff_emb = (question_diff_emb * weight.unsqueeze(-1)).sum(-2)
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        concept_que_diff_emb = self.W(torch.cat((concept_emb, question_diff_emb), dim=-1))

        interaction_emb = self.dropout_layer(concept_que_diff_emb) + correctness_emb
        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb[:, :-1])
        predict_layer_input = torch.cat((latent, self.dropout_layer(concept_que_diff_emb[:, 1:])), dim=-1)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        
        question_diff_seq = batch["question_diff_seq"][:, seq_start-1:] / num_question_diff
        similar_question_diff_seq = batch["similar_question_diff_seq"][:, seq_start-1:] / num_question_diff
        concept_hat_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["similar_question_seq"])
        question_hat_diff_emb = embed_question_diff(q2diff_transfer_table[batch["similar_question_diff_seq"]])
        weight_hat = q2diff_weight_table[batch["similar_question_diff_seq"]]
        question_hat_diff_emb = (question_hat_diff_emb * weight_hat.unsqueeze(-1)).sum(-2)
        concept_que_hat_diff_emb = self.W(torch.cat((concept_hat_emb, question_hat_diff_emb), dim=-1))
        predict_layer_input_hat = torch.cat((latent, self.dropout_layer(concept_que_hat_diff_emb[:, 1:])), dim=-1)
        predict_score_batch_hat = self.predict_layer(predict_layer_input_hat).squeeze(dim=-1)
        qdckt_loss_mask = torch.logical_and(mask_seq, torch.ne(batch["similar_question_mask_seq"], 0))
        qdckt_loss_all = torch.abs((predict_score_batch - predict_score_batch_hat)[:, seq_start-2:] - (similar_question_diff_seq - question_diff_seq))
        
        num_sample2 = torch.sum(qdckt_loss_mask[:, seq_start-1:]).item()
        qdckt_loss = torch.masked_select(qdckt_loss_all, qdckt_loss_mask[:, seq_start-1:]).sum() / num_sample2
        loss = predict_loss + w_qdckt_loss * qdckt_loss
        num_sample1 = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample1,
                    "num_sample": num_sample1
                },
                "qdckt loss": {
                    "value": qdckt_loss.detach().cpu().item() * num_sample2,
                    "num_sample": num_sample2
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        

    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass