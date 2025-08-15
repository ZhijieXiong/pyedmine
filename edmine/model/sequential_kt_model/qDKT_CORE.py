import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.registry import register_model

MODEL_NAME = "qDKT_CORE"


@register_model(MODEL_NAME)
class qDKT_CORE(nn.Module, DLSequentialKTModel):
    model_name = MODEL_NAME

    def __init__(self, params, objects):
        super(qDKT_CORE, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"][MODEL_NAME]
        embed_config = model_config["embed_config"]
        dim_concept = embed_config["concept"]["dim_item"]
        dim_question = embed_config["question"]["dim_item"]
        dim_correctness = embed_config["correctness"]["dim_item"]
        dim_emb = dim_concept + dim_question + dim_correctness
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
        self.question_net = nn.Sequential(
            nn.Linear(dim_question + dim_concept, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.user_net = nn.Linear(dim_latent, 2)
        self.softmax = nn.Softmax(-1)
        self.constant = nn.Parameter(torch.tensor(0.0))
        
    def get_qc_emb4single_concept(self, batch):
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]

        concept_question_emb = self.embed_layer.get_emb_concatenated(
            ("concept", "question"), (concept_seq, question_seq)
        )

        return concept_question_emb

    def get_qc_emb4only_question(self, batch):
        return self.embed_layer.get_emb_question_with_concept_fused(batch["question_seq"], fusion_type="mean")

    def forward(self, batch):        
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        question_emb = self.embed_layer.get_emb("question", batch["question_seq"])
        qc_emb = torch.cat((question_emb, concept_emb), dim=-1)
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        interaction_emb = torch.cat((qc_emb[:, :-1], correctness_emb[:, :-1]), dim=2)

        self.encoder_layer.flatten_parameters()
        latent, _ = self.encoder_layer(interaction_emb)

        logits = self.predict_layer(torch.cat([latent, qc_emb[:, 1:]], dim=-1))
        q_logits = self.question_net(qc_emb[:, 1:].detach())
        s_logits = self.user_net(latent.detach())

        z_QKS = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=True, S_fact=True)
        z_Q = self.fusion(logits, q_logits, s_logits, Q_fact=True, K_fact=False, S_fact=False)
        logit_core = z_QKS - z_Q

        # TIE
        z_nde = self.fusion(logits.clone().detach(), q_logits.clone().detach(), s_logits.clone().detach(),
                            Q_fact=True, K_fact=False, S_fact=False)
        # NDE = z_Q - z
        mask_bool_seq_ = batch["mask_seq"][:, 1:].unsqueeze(-1).bool()
        z_nde_pred = torch.masked_select(z_nde, mask_bool_seq_).view(-1, 2)
        q_pred = torch.masked_select(q_logits, mask_bool_seq_).view(-1, 2)
        z_qks_pred = torch.masked_select(z_QKS, mask_bool_seq_).view(-1, 2)

        return z_nde_pred, q_pred, z_qks_pred, logit_core

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        _, _, _, logit_core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_core, dim=-1)[:, :, 1]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start - 2:], mask_bool_seq[:, seq_start - 1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        z_nde_pred, q_pred, z_qks_pred, logit_core = self.forward(batch)
        predict_score_batch = torch.softmax(logit_core, dim=-1)[:, :, 1]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start - 2:], mask_bool_seq[:, seq_start - 1:])
        ground_truth = torch.masked_select(batch["correctness_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = torch.nn.functional.cross_entropy(z_qks_pred, ground_truth) + \
                       torch.nn.functional.cross_entropy(q_pred, ground_truth)
        p_te = self.softmax(z_qks_pred).clone().detach()
        KL_loss = - p_te * self.softmax(z_nde_pred).log()
        KL_loss = KL_loss.sum(1).mean()
        loss = predict_loss + KL_loss

        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "KL loss": {
                    "value": KL_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def fusion(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"][MODEL_NAME]["fusion_mode"]
        predict_K, predict_Q, predict_S = self.transform(predict_K, predict_Q, predict_S, Q_fact, K_fact, S_fact)

        if fusion_mode == 'rubin':
            # 鲁宾因果模型（潜在因果框架）
            z = predict_K * torch.sigmoid(predict_Q)

        elif fusion_mode == 'hm':
            #
            z = predict_K * predict_S * predict_Q
            z = torch.log(z + 1e-12) - torch.log1p(z)

        elif fusion_mode == 'sum':
            z = predict_K + predict_Q + predict_S
            z = torch.log(torch.sigmoid(z) + 1e-12)

        else:
            raise NotImplementedError()

        return z

    def transform(self, predict_K, predict_Q, predict_S, Q_fact=False, K_fact=False, S_fact=False):
        fusion_mode = self.params["models_config"][MODEL_NAME]["fusion_mode"]
        if not K_fact:
            predict_K = self.constant * torch.ones_like(predict_K).to(self.params["device"])
        if not Q_fact:
            predict_Q = self.constant * torch.ones_like(predict_Q).to(self.params["device"])
        if not S_fact:
            predict_S = self.constant * torch.ones_like(predict_S).to(self.params["device"])

        if fusion_mode == 'hm':
            predict_K = torch.sigmoid(predict_K)
            predict_Q = torch.sigmoid(predict_Q)
            predict_S = torch.sigmoid(predict_S)

        return predict_K, predict_Q, predict_S

    def get_knowledge_state(self, batch, only_last_state=True):
        pass
    