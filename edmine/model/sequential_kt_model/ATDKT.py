import torch
import torch.nn.functional as F
from torch import nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.loss import binary_cross_entropy


def ut_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


class ATDKT(nn.Module):
    model_name = "ATDKT"

    def __init__(self, params, objects):
        super(ATDKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["ATDKT"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        dim_emb = model_config["dim_emb"]
        dim_latent = model_config["dim_latent"]
        num_rnn_layer = model_config["num_rnn_layer"]
        dropout = model_config["dropout"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.dkt_encoder = nn.GRU(dim_emb * 2, dim_latent, batch_first=True, num_layers=num_rnn_layer)
        self.dkt_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_emb + dim_latent, (dim_latent + dim_emb) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((dim_latent + dim_emb) // 2, 1),
            nn.Sigmoid()
        )
        self.QT_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_emb, num_concept),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_concept, num_concept),
            nn.Sigmoid()
        )
        self.IK_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_latent, dim_latent // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_latent // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        question_seq = batch["question_seq"]
        correctness_seq = batch["correctness_seq"]

        question_emb = self.embed_layer.get_emb("question", question_seq)
        correctness_emb = self.embed_layer.get_emb("correctness", correctness_seq)

        # predict right or wrong
        interaction_emb = torch.cat((question_emb, correctness_emb), dim=-1)[:, :-1]
        latent, _ = self.dkt_encoder(interaction_emb)
        KT_predict_score = self.dkt_classifier(torch.cat((latent, question_emb[:, 1:]), dim=-1)).squeeze(-1)

        # predict student's history accuracy
        IK_predict_score = self.IK_predictor(latent).squeeze(-1)

        return KT_predict_score, IK_predict_score

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation
        predict_score_batch = self.forward(batch)[0]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        model_config = self.params["models_config"]["ATDKT"]
        IK_start = model_config["IK_start"]
        mask_seq = torch.ne(batch["mask_seq"], 0)
        question_id = torch.unique(batch["question_seq"])
        KT_ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        IK_ground_truth = batch["history_acc_seq"][:, IK_start + 1:][mask_seq[:, IK_start + 1:]]
        QT_ground_truth = self.objects["dataset"]["q_table_tensor"][question_id]
        
        KT_predict_score_batch, IK_predict_score_batch = self.forward(batch)
        
        KT_predict_score = torch.masked_select(KT_predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        KT_predict_loss = binary_cross_entropy(KT_predict_score, KT_ground_truth, self.params["device"])
        
        IK_predict_score = IK_predict_score_batch[:, IK_start:][mask_seq[:, IK_start + 1:]]
        IK_predict_loss = F.mse_loss(IK_predict_score, IK_ground_truth)
        
        question_emb = self.embed_layer.get_emb("question", question_id)
        QT_predict_score = self.QT_classifier(question_emb)
        QT_predict_loss = F.multilabel_soft_margin_loss(QT_predict_score, QT_ground_truth)

        loss = (KT_predict_loss + QT_predict_loss * self.params["loss_config"]["QT loss"] +
                IK_predict_loss * self.params["loss_config"]["IK loss"])

        num_sample_KT = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        num_sample_QT = question_id.shape[0]
        num_sample_IK = torch.sum(batch["mask_seq"][:, IK_start + 1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": KT_predict_loss.detach().cpu().item() * num_sample_KT,
                    "num_sample": num_sample_KT
                },
                "QT loss": {
                    "value": QT_predict_loss.detach().cpu().item() * num_sample_QT,
                    "num_sample": num_sample_QT
                },
                "IK loss": {
                    "value": IK_predict_loss.detach().cpu().item() * num_sample_IK,
                    "num_sample": num_sample_IK
                }
            },
            "predict_score": KT_predict_score_batch[mask_seq[:, seq_start-1:]],
            "predict_score_batch": KT_predict_score_batch
        }

    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass