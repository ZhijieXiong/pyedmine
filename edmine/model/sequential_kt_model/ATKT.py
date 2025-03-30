import torch
import numpy as np
import torch.nn as nn

from torch.nn.functional import one_hot
from torch.autograd import Variable, grad

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import binary_cross_entropy


def l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    else:
        d = d.cpu().numpy()
    # "learning_rate": 0.01,"lr_schedule_step": 30,"lr_schedule_gamma": 0.5
    d = d / (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


class ATKT(nn.Module, DLSequentialKTModel):
    model_name = "ATKT"

    def __init__(self, params, objects):
        super(ATKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["ATKT"]
        dim_concept = model_config["embed_config"]["concept"]["dim_item"]
        dim_latent = model_config["dim_latent"]
        dim_attention = model_config["dim_attention"]
        dim_correctness = model_config["embed_config"]["correctness"]["dim_item"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.mlp = nn.Linear(dim_latent, dim_attention)
        self.similarity = nn.Linear(dim_attention, 1, bias=False)
        self.rnn = nn.GRU(dim_concept + dim_correctness, dim_latent, batch_first=True)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def attention_module(self, rnn_output):
        att_w = self.mlp(rnn_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)

        # 这一步导致数据泄露！！！计算softmax时没有屏蔽未来的数据
        # (bs, seq_len, 1) -> (bs, seq_len, 1)
        # alphas = nn.Softmax(dim=1)(att_w)
        # attn_output = att_w * rnn_output

        # pykt修改后的代码
        seq_len = rnn_output.shape[1]
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(self.params["device"])
        att_w = att_w.transpose(1, 2).expand(rnn_output.shape[0], rnn_output.shape[1], rnn_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        alphas = torch.nn.functional.softmax(att_w, dim=-1)
        attn_output = torch.bmm(alphas, rnn_output)

        attn_output_cum = torch.cumsum(attn_output, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_output
        final_output = torch.cat((attn_output_cum_1, rnn_output), 2)

        return final_output

    def get_interaction_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        correctness_emb = self.embed_layer.get_emb("correctness", batch["correctness_seq"])
        concept_correct_emb = torch.cat((concept_emb, correctness_emb), 2)
        correct_concept_emb = torch.cat((correctness_emb, concept_emb), 2)
        correctness_seq = batch["correctness_seq"].unsqueeze(2).expand_as(concept_correct_emb)
        interaction_emb = torch.where(correctness_seq == 1, concept_correct_emb, correct_concept_emb)
        return interaction_emb
    
    def get_latent(self, interaction_emb):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(interaction_emb)
        latent = self.attention_module(rnn_out)
        return latent

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        interaction_emb = self.get_interaction_emb(batch)
        latent = self.get_latent(interaction_emb)
        predict_layer_input = torch.cat((latent[:,:-1], concept_emb[:, 1:]), dim=-1)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
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

    def get_predict_loss(self, batch, seq_start=2):
        model_config = self.params["models_config"]["ATKT"]
        epsilon = model_config["epsilon"]
        w_adv_loss = self.params["loss_config"]["adv loss"]
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        mask_seq = torch.ne(batch["mask_seq"], 0)
        correctness_seq = batch["correctness_seq"]
        ground_truth = torch.masked_select(correctness_seq[:, seq_start-1:].long(), mask_seq[:, seq_start-1:])

        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        interaction_emb = self.get_interaction_emb(batch)
        latent = self.get_latent(interaction_emb)
        predict_layer_input = torch.cat((latent[:,:-1], concept_emb[:, 1:]), dim=-1)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        loss = 0.
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        loss += predict_loss

        interaction_grad = grad(predict_loss, interaction_emb, retain_graph=True)
        perturbation = torch.FloatTensor(epsilon * l2_normalize_adv(interaction_grad[0].data))
        perturbation = Variable(perturbation).to(self.params["device"])
        adv_interaction_emb = self.get_interaction_emb(batch) + perturbation
        adv_latent = self.get_latent(adv_interaction_emb)
        adv_predict_layer_input = torch.cat((adv_latent[:,:-1], concept_emb[:, 1:]), dim=-1)
        adv_predict_score_batch = self.predict_layer(adv_predict_layer_input).squeeze(dim=-1)
        adv_predict_score = torch.masked_select(adv_predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        adv_loss = binary_cross_entropy(adv_predict_score, ground_truth, self.params["device"])
        loss += adv_loss * w_adv_loss

        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "adv loss": {
                    "value": adv_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
    
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        num_question = target_question.shape[1]
        batch_size = batch["correctness_seq"].shape[0]
        
        interaction_emb = self.get_interaction_emb(batch)
        latent = self.get_latent(interaction_emb)
        target_latent = latent[:, target_index-1]
        target_question_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, target_question)
        target_latent_extend = target_latent.repeat_interleave(num_question, dim=0).view(batch_size, num_question, -1)
        predict_layer_input = torch.cat((target_latent_extend, target_question_emb), dim=2)
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        return predict_score

    def get_knowledge_state(self, batch):
        pass
