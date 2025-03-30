import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.loss import binary_cross_entropy


class MLP(nn.Module):
    def __init__(self, num_layer, dim_in, dim_out, dropout):
        super().__init__()

        self.linear_list = nn.ModuleList([
            nn.Linear(dim_in, dim_in)
            for _ in range(num_layer)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(dim_in, dim_out)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.linear_list:
            x = torch.relu(lin(x))
        return self.out(self.dropout(x))


def sigmoid_inverse(x, epsilon=1e-8):
    return torch.log(x / (1 - x + epsilon) + epsilon)


class QIKT(nn.Module, DLSequentialKTModel):
    model_name = "QIKT"

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["QIKT"]
        num_question = model_config["embed_config"]["question"]["num_item"]
        num_concept = model_config["embed_config"]["concept"]["num_item"]
        dim_emb = model_config["embed_config"]["concept"]["dim_item"]
        rnn_type = model_config["rnn_type"]
        num_rnn_layer = model_config["num_rnn_layer"]
        num_mlp_layer = model_config["num_mlp_layer"]
        dropout = model_config["dropout"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        if rnn_type == "rnn":
            self.rnn_layer4question = nn.RNN(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
            self.rnn_layer4concept = nn.RNN(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        elif rnn_type == "lstm":
            self.rnn_layer4question = nn.LSTM(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
            self.rnn_layer4concept = nn.LSTM(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        else:
            self.rnn_layer4question = nn.GRU(dim_emb * 4, dim_emb, batch_first=True, num_layers=num_rnn_layer)
            self.rnn_layer4concept = nn.GRU(dim_emb * 2, dim_emb, batch_first=True, num_layers=num_rnn_layer)
        self.dropout_layer = nn.Dropout(dropout)
        self.predict_layer4q_next = MLP(num_mlp_layer, dim_emb * 3, 1, dropout)
        self.predict_layer4q_all = MLP(num_mlp_layer, dim_emb, num_question, dropout)
        self.predict_layer4c_next = MLP(num_mlp_layer, dim_emb * 3, num_concept, dropout)
        self.predict_layer4c_all = MLP(num_mlp_layer, dim_emb, num_concept, dropout)
        self.que_discrimination_layer = MLP(num_mlp_layer, dim_emb * 2, 1, dropout)

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        model_config = self.params["models_config"]["QIKT"]
        lambda_q_all = model_config["lambda_q_all"]
        lambda_c_next = model_config["lambda_c_next"]
        lambda_c_all = model_config["lambda_c_all"]
        use_irt = model_config["use_irt"]
        dim_emb = model_config["embed_config"]["concept"]["dim_item"]

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        question_emb = self.embed_layer.get_emb("question", batch["question_seq"])
        qc_emb = torch.cat((question_emb, concept_emb), dim=-1)
        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        qca_emb = torch.cat([
            qc_emb.mul((1 - batch["correctness_seq"]).unsqueeze(-1).repeat(1, 1, dim_emb * 2)),
            qc_emb.mul(batch["correctness_seq"].unsqueeze(-1).repeat(1, 1, dim_emb * 2))
        ],
            dim=-1
        )
        ca_emb = torch.cat([
            concept_emb.mul((1 - batch["correctness_seq"]).unsqueeze(-1).repeat(1, 1, dim_emb)),
            concept_emb.mul(batch["correctness_seq"].unsqueeze(-1).repeat(1, 1, dim_emb))
        ],
            dim=-1
        )

        self.rnn_layer4question.flatten_parameters()
        self.rnn_layer4concept.flatten_parameters()
        latent_question = self.dropout_layer(self.rnn_layer4question(qca_emb[:, :-1])[0])
        latent_concept = self.dropout_layer(self.rnn_layer4concept(ca_emb[:, :-1])[0])

        predict_score_q_next = torch.sigmoid(self.predict_layer4q_next(
            torch.cat((qc_emb[:, 1:], latent_question), dim=-1)
        )).squeeze(-1)
        predict_score_q_all = torch.sigmoid(self.predict_layer4q_all(latent_question))
        predict_score_c_next = torch.sigmoid(self.predict_layer4c_next(
            torch.cat((qc_emb[:, 1:], latent_concept), dim=-1)
        ))
        predict_score_c_all = torch.sigmoid(self.predict_layer4c_all(latent_concept))

        predict_score_q_all = torch.gather(predict_score_q_all, 2, batch["question_seq"].unsqueeze(-1)[:, 1:]).squeeze(-1)

        # predict_score_c_next和predict_score_c_all原代码没写怎么处理一道习题对应多个知识点，我的理解是各个知识点上的分数取平均值
        q2c_seq = q2c_transfer_table[batch["question_seq"]]
        q2c_mask_seq = q2c_mask_table[batch["question_seq"]]
        num_max_concept = q2c_seq.shape[-1]

        # [bs, seq_len, num_max_c] => [bs, seq_len, num_max_c, 1]
        q2c_seq = q2c_seq[:, 1:].unsqueeze(-1)
        # [bs, seq_len, num_max_c] => [bs, seq_len, num_max_c, 1]
        q2c_mask_seq_repeated = q2c_mask_seq[:, 1:].unsqueeze(-1)
        # [bs, seq_len, num_max_c] => [bs, seq_len]
        q2c_mask_seq_sum = q2c_mask_seq[:, 1:].sum(dim=-1)

        # [bs, seq_len, num_c] => [bs, seq_len, num_max_c, num_c]
        predict_score_c_next = predict_score_c_next.unsqueeze(-2).repeat(1, 1, num_max_concept, 1)
        predict_score_c_next = torch.gather(predict_score_c_next, -1, q2c_seq) * q2c_mask_seq_repeated
        predict_score_c_next = predict_score_c_next.squeeze(-1).sum(dim=-1) / q2c_mask_seq_sum

        predict_score_c_all = predict_score_c_all.unsqueeze(-2).repeat(1, 1, num_max_concept, 1)
        predict_score_c_all = torch.gather(predict_score_c_all, -1, q2c_seq) * q2c_mask_seq_repeated
        predict_score_c_all = predict_score_c_all.squeeze(-1).sum(dim=-1) / q2c_mask_seq_sum

        if use_irt:
            predict_score = (sigmoid_inverse(predict_score_q_all) * lambda_q_all +
                             sigmoid_inverse(predict_score_c_all) * lambda_c_all +
                             sigmoid_inverse(predict_score_c_next) * lambda_c_next)
            predict_score = torch.sigmoid(predict_score)
        else:
            predict_score = (predict_score_q_all * lambda_q_all +
                             predict_score_c_all * lambda_c_all +
                             predict_score_c_next * lambda_c_next)
            predict_score = predict_score / (lambda_q_all + lambda_c_all + lambda_c_next)

        return predict_score, predict_score_q_next, predict_score_q_all, predict_score_c_next, predict_score_c_all

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch, _, _, _, _ = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        loss_wight = self.params["loss_config"]
        use_irt = self.params["models_config"]["QIKT"]["use_irt"]
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score_batch, predict_score_q_next, predict_score_q_all, predict_score_c_next, predict_score_c_all = (
            self.forward(batch))
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])
        predict_score_q_next = torch.masked_select(predict_score_q_next[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])
        predict_score_q_all = torch.masked_select(predict_score_q_all[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])
        predict_score_c_next = torch.masked_select(predict_score_c_next[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])
        predict_score_c_all = torch.masked_select(predict_score_c_all[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_bool_seq[:, seq_start-1:])

        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        predict_loss_q_next = binary_cross_entropy(predict_score_q_next, ground_truth, self.params["device"])
        predict_loss_q_all = binary_cross_entropy(predict_score_q_all, ground_truth, self.params["device"])
        predict_loss_c_next = binary_cross_entropy(predict_score_c_next, ground_truth, self.params["device"])
        predict_loss_c_all = binary_cross_entropy(predict_score_c_all, ground_truth, self.params["device"])


        loss = (predict_loss + loss_wight["q all loss"] * predict_loss_q_all +
                loss_wight["c all loss"] * predict_loss_c_all + loss_wight["c next loss"] * predict_loss_c_next)
        if not use_irt:
            loss = loss + loss_wight["q next loss"] * predict_loss_q_next

        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "q all loss": {
                    "value": predict_loss_q_all.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "q next loss": {
                    "value": predict_loss_q_next.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "c all loss": {
                    "value": predict_loss_c_all.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "c next loss": {
                    "value": predict_loss_c_next.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass
