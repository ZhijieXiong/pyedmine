import numpy as np
import torch
import torch.nn as nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer


class HawkesKT(nn.Module, DLSequentialKTModel):
    model_name = "HawkesKT"
    
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["HawkesKT"]
        
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        for _, module in self.embed_layer.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def get_predict_score(self, batch, seq_start=2):
        model_config = self.params["models_config"]["HawkesKT"]
        time_log = model_config["time_log"]
        _, num_concept = self.objects["dataset"]["q_table"].shape
        
        concept_seq = batch["concept_seq"]
        question_seq = batch["question_seq"]
        correctness_seq = batch["correctness_seq"]
        batch_size, seq_len = concept_seq.shape
        inters = concept_seq + correctness_seq * num_concept
        
        alpha_src_emb = self.embed_layer.get_emb("alpha_interaction", inters)
        alpha_target_emb = self.embed_layer.get_emb("alpha_concept", concept_seq)
        
        beta_src_emb = self.embed_layer.get_emb("beta_interaction", inters)
        beta_target_emb = self.embed_layer.get_emb("beta_concept", concept_seq)
        
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))
        betas = torch.clamp(betas + 1, min=0, max=10)

        if "time_seq" in batch:
            time_seq = batch["time_seq"].double()
            delta_t = (time_seq[:, :, None] - time_seq[:, None, :]).abs().double()
        else:
            delta_t = torch.ones(batch_size, seq_len, seq_len).double().to(self.params["device"])
        delta_t = torch.log(delta_t + 1e-10) / np.log(time_log)
        cross_effects = alphas * torch.exp(-betas * delta_t)
        valid_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1)
        mask = (torch.from_numpy(valid_mask) == 0).to(self.params["device"])
        sum_t = cross_effects.masked_fill(mask, 0).sum(-2)

        problem_bias = self.embed_layer.get_emb("question", question_seq).squeeze(dim=-1)
        skill_bias = self.embed_layer.get_emb("concept", concept_seq).squeeze(dim=-1)

        predict_score_batch = (problem_bias + skill_bias + sum_t).sigmoid()[:, 1:]
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_knowledge_state(self, batch):
        pass