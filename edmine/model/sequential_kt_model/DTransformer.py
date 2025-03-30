import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.MultiHeadAttention import MultiHeadAttention4Dtransformer


class TransformerLayer4Dtransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        model_config = self.params["models_config"]["DTransformer"]
        dim_model = model_config["dim_model"]
        dropout = model_config["dropout"]

        self.masked_attn_head = MultiHeadAttention4Dtransformer(params)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, query, key, values, seqs_length, peek_cur=False):
        model_config = self.params["models_config"]["DTransformer"]
        dropout = model_config["dropout"]

        # construct mask
        seq_len = query.size(1)
        mask = torch.ones(seq_len, seq_len).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.params["device"])

        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                # sample for each batch
                if seqs_length[b] < DTransformer.MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(seqs_length[b] - 1), max(1, int(seqs_length[b] * dropout))
                )
                for i in idx:
                    mask[b, :, i + 1:, i] = 0

        # apply transformer layer
        query_, scores = self.masked_attn_head(
            query, key, values, mask, max_out=not peek_cur
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class DTransformer(nn.Module, DLSequentialKTModel):
    model_name = "DTransformer"
    MIN_SEQ_LEN = 5

    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["DTransformer"]
        dim_model = model_config["dim_model"]
        dim_final_fc = model_config["dim_final_fc"]
        num_know = model_config["num_know"]
        dropout = model_config["dropout"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        # 前三个都是AKT中的
        # 提取习题表征（习题embedding作为k、q、v）
        self.question_encoder = TransformerLayer4Dtransformer(params)
        # 提取交互表征（交互，即习题和回答结果，的embedding作为k、q、v）
        self.knowledge_encoder = TransformerLayer4Dtransformer(params)
        # 提取知识状态（习题表征作为k和q，交互表征作为v）
        self.knowledge_retriever = TransformerLayer4Dtransformer(params)
        params_ = deepcopy(params)
        params_["models_config"]["DTransformer"]["key_query_same"] = False
        self.block4 = TransformerLayer4Dtransformer(params_)

        self.num_know = num_know
        self.knowledge_params = nn.Parameter(torch.empty(num_know, dim_model))
        torch.nn.init.uniform_(self.knowledge_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(dim_model * 2, dim_final_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc, dim_final_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_final_fc // 2, 1),
        )

    def forward(self, concept_emb, interaction_emb, seqs_length):
        num_head = self.params["models_config"]["DTransformer"]["num_head"]

        # 融合了习题难度的concept embedding作为k、q、v提取有上下文信息的question embedding（论文中的m）
        question_representation, _ = self.question_encoder(concept_emb, concept_emb, concept_emb, seqs_length,
                                                            peek_cur=True)
        # interaction embedding作为k、q、v提取有上下文信息的interaction embedding
        interaction_representation, _ = self.knowledge_encoder(interaction_emb, interaction_emb, interaction_emb,
                                                                seqs_length, peek_cur=True)
        # question embedding作为k、q，interaction embedding作为v，提取学生的知识状态（AKT），该状态经过read-out得到论文的z
        knowledge_representation, q_scores = self.knowledge_retriever(question_representation,
                                                                        question_representation,
                                                                        interaction_representation,
                                                                        seqs_length, peek_cur=True)

        batch_size, seq_len, dim_model = knowledge_representation.size()
        num_prototype = self.num_know

        # query是论文中的K，作为attention的q。该模型用一个(num_prototype, dim_model)的tensor来表征底层的知识状态
        K = (
            self.knowledge_params[None, :, None, :]
            .expand(batch_size, -1, seq_len, -1)
            .contiguous()
            .view(batch_size * num_prototype, seq_len, dim_model)
        )
        question_representation = question_representation.unsqueeze(1).expand(-1, num_prototype, -1, -1).reshape_as(K)
        knowledge_representation = knowledge_representation.unsqueeze(1).expand(-1, num_prototype, -1, -1).reshape_as(K)

        # 论文中的z和attention的score（有MaxOut的attention）
        z, k_scores = self.block4(
            K, question_representation, knowledge_representation, torch.repeat_interleave(seqs_length, num_prototype),
            peek_cur=False
        )
        z = (
            z.view(batch_size, num_prototype, seq_len, dim_model)  # unpack dimensions
            .transpose(1, 2)  # (batch_size, seq_len, num_prototype, dim_model)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        k_scores = (
            k_scores.view(batch_size, num_prototype, num_head, seq_len, seq_len)  # unpack dimensions
            .permute(0, 2, 3, 1, 4)  # (batch_size, n_heads, seq_len, num_prototype, seq_len)
            .contiguous()
        )
        return z, q_scores, k_scores

    def get_latent(self, batch):
        concept_emb, _, _ = self.embed_input(batch)
        z = self.get_z(batch)
        query = concept_emb[:, :, :]
        # predict T+N，即论文中公式(19)的z_{q_t}
        latent = self.readout(z[:, : query.size(1), :], query)

        return latent

    def get_predict_score(self, batch, seq_start=2):
        concept_seq = batch["concept_seq"]
        correctness_seq = batch["correctness_seq"]
        question_seq = batch["question_seq"]

        # concept_emb是融合了习题难度的concept，所以实际上可以看做question的embedding
        seqs_length = (correctness_seq >= 0).sum(dim=1)
        concept_emb, interaction_emb, _ = self.embed_input({
            "concept_seq": concept_seq, "correctness_seq": correctness_seq, "question_seq": question_seq
        })
        z, _, _ = self.forward(concept_emb, interaction_emb, seqs_length)

        n = 1
        query = concept_emb[:, n - 1:, :]
        # predict T+N，即论文中公式(19)的z_{q_t}
        latent = self.readout(z[:, : query.size(1), :], query)

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = torch.sigmoid(self.out(torch.cat([query, latent], dim=-1)).squeeze(-1))[:, 1:]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch):
        loss_result = self.get_predict_loss_(batch)
        predict_loss = loss_result["predict loss"]
        reg_loss = loss_result["reg loss"]

        correctness_seq = batch["correctness_seq"]
        seqs_length = (correctness_seq >= 0).sum(dim=1)
        min_len = seqs_length.min().item()
        num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
        weight_reg = self.params["loss_config"]["reg loss"]
        weight_cl_loss = self.params["loss_config"]["cl loss"]
        if min_len < DTransformer.MIN_SEQ_LEN:
            # skip CL for batches that are too short
            return {
                "total_loss": predict_loss + reg_loss * weight_reg,
                "losses_value": {
                    "predict loss": {
                        "value": predict_loss.detach().cpu().item() * num_sample,
                        "num_sample": num_sample
                    },
                    "reg loss": {
                        "value": reg_loss.detach().cpu().item() * num_sample,
                        "num_sample": num_sample
                    },
                    "cl loss": {
                        "value": 0,
                        "num_sample": 1
                    }
                },
                "predict_score": loss_result["predict_score"],
                "predict_score_batch": loss_result["predict_score_batch"]
            }
        else:
            cl_loss = self.get_cl_loss(batch)
            return {
                "total_loss": predict_loss + reg_loss * weight_reg + cl_loss * weight_cl_loss,
                "losses_value": {
                    "predict loss": {
                        "value": predict_loss.detach().cpu().item() * num_sample,
                        "num_sample": num_sample
                    },
                    "reg loss": {
                        "value": reg_loss.detach().cpu().item() * num_sample,
                        "num_sample": num_sample
                    },
                    "cl loss": {
                        "value": cl_loss.detach().cpu().item(),
                        "num_sample": 1
                    }
                },
                "predict_score": loss_result["predict_score"],
                "predict_score_batch": loss_result["predict_score_batch"]
            }

    def get_z(self, batch):
        correctness_seq = batch["correctness_seq"]
        seqs_length = (correctness_seq >= 0).sum(dim=1)
        concept_emb, interaction_emb, _ = self.embed_input(batch)
        z, _, _ = self.forward(concept_emb, interaction_emb, seqs_length)

        return z

    def embed_input(self, batch):
        concept_seq = batch["concept_seq"]
        correctness_seq = batch["correctness_seq"]
        question_seq = batch["question_seq"]

        # set prediction mask
        concept_seq = concept_seq.masked_fill(concept_seq < 0, 0)
        correctness_seq = correctness_seq.masked_fill(correctness_seq < 0, 0)

        concept_emb = self.embed_layer.get_emb("concept", concept_seq)
        interaction_emb = self.embed_layer.get_emb("correctness", correctness_seq) + concept_emb
        
        question_diff_emb = 0.0
        question_seq = question_seq.masked_fill(question_seq < 0, 0)
        question_diff_emb = self.embed_layer.get_emb("question_difficulty", question_seq)
        concept_variation_emb = self.embed_layer.get_emb("concept_variation", concept_seq)
        # AKT question encoder 输入
        concept_emb = concept_emb + concept_variation_emb * question_diff_emb
        correctness_variation_emb = self.embed_layer.get_emb("correctness_variation", correctness_seq) + concept_variation_emb
        # AKT knowledge encoder 输入
        interaction_emb = interaction_emb + correctness_variation_emb * question_diff_emb

        return concept_emb, interaction_emb, question_diff_emb

    def readout(self, z, concept_emb):
        # 使用当前时刻的concept作为q，学习到的K作为k，提取的知识状态z作为v，做attention，得到融合后的知识状态
        batch_size, seq_len, _ = concept_emb.size()
        key = (
            self.knowledge_params[None, None, :, :]
            .expand(batch_size, seq_len, -1, -1)
            .view(batch_size * seq_len, self.num_know, -1)
        )
        value = z.reshape(batch_size * seq_len, self.num_know, -1)

        beta = torch.matmul(
            key,
            concept_emb.reshape(batch_size * seq_len, -1, 1),
        ).view(batch_size * seq_len, 1, self.num_know)
        alpha = torch.softmax(beta, -1)
        # 论文公式(19)
        return torch.matmul(alpha, value).view(batch_size, seq_len, -1)

    def predict(self, concept_seq, correctness_seq, question_seq, n=1):
        # concept_emb是融合了习题难度的concept，所以实际上可以看做question的embedding
        seqs_length = (correctness_seq >= 0).sum(dim=1)
        concept_emb, interaction_emb, question_difficulty = self.embed_input({
            "concept_seq": concept_seq, "correctness_seq": correctness_seq, "question_seq": question_seq
        })
        z, q_scores, k_scores = self(concept_emb, interaction_emb, seqs_length)

        query = concept_emb[:, n - 1:, :]
        # predict T+N，即论文中公式(19)的z_{q_t}
        latent = self.readout(z[:, : query.size(1), :], query)

        predict_logits = self.out(torch.cat([query, latent], dim=-1)).squeeze(-1)

        reg_loss = (question_difficulty ** 2).mean()

        return predict_logits, z, concept_emb, reg_loss, (q_scores, k_scores)

    def get_predict_loss_(self, batch):
        window = self.params["models_config"]["DTransformer"]["window"]

        concept_seq = batch["concept_seq"]
        correctness_seq = batch["correctness_seq"]
        question_seq = batch["question_seq"]

        # reg_loss实际上就是question difficulty embedding的二范数，和AKT一样，作为loss的一部分
        predict_logits_batch, z, concept_emb, reg_loss, _ = self.predict(concept_seq, correctness_seq, question_seq)
        ground_truth = correctness_seq[correctness_seq >= 0].float()
        predict_logits = predict_logits_batch[correctness_seq >= 0]
        # binary_cross_entropy_with_logits = Sigmoid + BCE loss，因此predict_logits是任意数
        predict_loss = F.binary_cross_entropy_with_logits(predict_logits, ground_truth, reduction="mean")
        for i in range(1, window):
            label = correctness_seq[:, i:]
            query = concept_emb[:, i:, :]
            h = self.readout(z[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            predict_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )

        predict_loss /= window

        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = torch.sigmoid(predict_logits_batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch, mask_bool_seq[:, 1:])

        return {
            "predict loss": predict_loss,
            "reg loss": reg_loss,
            "predict_score_batch": predict_score_batch,
            "predict_score": predict_score
        }

    def get_cl_loss(self, batch):
        model_config = self.params["models_config"]["DTransformer"]
        dropout = model_config["dropout"]

        concept_seq = batch["concept_seq"]
        correctness_seq = batch["correctness_seq"]
        question_seq = batch["question_seq"]

        batch_size = correctness_seq.size(0)
        seqs_length = (correctness_seq >= 0).sum(dim=1)
        min_len = seqs_length.min().item()

        # augmentation
        concept_seq_ = concept_seq.clone()
        correctness_seq_ = correctness_seq.clone()

        question_seq_ = None
        question_seq_ = question_seq.clone()

        # manipulate order
        for b in range(batch_size):
            idx = random.sample(
                range(seqs_length[b] - 1), max(1, int(seqs_length[b] * dropout))
            )
            for i in idx:
                concept_seq_[b, i], concept_seq_[b, i + 1] = concept_seq_[b, i + 1], concept_seq_[b, i]
                correctness_seq_[b, i], correctness_seq_[b, i + 1] = correctness_seq_[b, i + 1], correctness_seq_[b, i]
                question_seq_[b, i], question_seq_[b, i + 1] = question_seq_[b, i + 1], question_seq_[b, i]

        # hard negative
        s_flip = correctness_seq.clone()
        for b in range(batch_size):
            # manipulate score
            idx = random.sample(
                range(seqs_length[b]), max(1, int(seqs_length[b] * dropout))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]

        # z就是论文中的z_{q_t}
        z_1 = self.get_z(batch)
        z_2 = self.get_z({"concept_seq": concept_seq_, "correctness_seq": correctness_seq_, "question_seq": question_seq_})
        input_ = self.sim(z_1[:, :min_len, :], z_2[:, :min_len, :])
        z_3 = self.get_z({"concept_seq": concept_seq, "correctness_seq": s_flip, "question_seq": question_seq})
        hard_neg = self.sim(z_1[:, :min_len, :], z_3[:, :min_len, :])
        input_ = torch.cat((input_, hard_neg), dim=1)
        target = (
            torch.arange(correctness_seq.size(0))[:, None]
            .to(self.params["device"])
            .expand(-1, min_len)
        )
        cl_loss = F.cross_entropy(input_, target)

        return cl_loss

    def sim(self, z1, z2):
        temperature = self.params["models_config"]["DTransformer"]["temperature"]
        bs, seq_len, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seq_len, self.num_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seq_len, self.num_know, -1)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / temperature

    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass
