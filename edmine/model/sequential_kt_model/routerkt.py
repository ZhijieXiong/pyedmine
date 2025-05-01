import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import binary_cross_entropy

class MoHAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, n_shared_heads, 
                 n_selected_heads, dropout, kq_same,
                 seq_len=200, routing_mode="dynamic"):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.h_shared = n_shared_heads
        self.h_selected = n_selected_heads
        self.kq_same = kq_same
        self.routing_mode = routing_mode
        
        # Linear layers for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        if not kq_same:
            self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Routing networks
        if routing_mode == "dynamic":
            self.wg = nn.Linear(d_model, n_heads - n_shared_heads, bias=False)  # Router for dynamic heads
            
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
        # Track routing statistics for load balancing
        self.register_buffer('head_selections', torch.zeros(n_heads - n_shared_heads))
        self.register_buffer('head_routing_probs', torch.zeros(n_heads - n_shared_heads))
        
    def get_balance_loss(self):
        # Calculate load balance loss for dynamic heads
        f = self.head_selections / (self.head_selections.sum() + 1e-5)
        P = self.head_routing_probs / (self.head_routing_probs.sum() + 1e-5)
        balance_loss = (f * P).sum()
        return balance_loss
        
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        seq_len = q.size(1)
        
        # Linear projections
        q = self.q_linear(q)  # [bs, seq_len, d_model]
        if self.kq_same:
            k = q
        else:
            k = self.k_linear(k)
        v = self.v_linear(v)
        
        # Reshape for attention computation
        q = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)  # [bs, h, seq_len, d_k]
        k = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # [bs, h, seq_len, seq_len]
        
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # First position zero padding
        pad_zero = torch.zeros(bs, self.h, 1, scores.size(-1)).to(q.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        
        # Calculate routing scores for dynamic heads
        q_for_routing = q.permute(0, 2, 1, 3).reshape(bs * seq_len, self.h * self.d_k)  # [bs*seq_len, h*d_k]
        
        # Handle dynamic heads routing
        if self.routing_mode == "dynamic":
            # Use learned routing weights
            logits = self.wg(q_for_routing)  # [bs*seq_len, n_dynamic_heads]
            gates = F.softmax(logits, dim=1)  # [bs*seq_len, n_dynamic_heads]
        else:  # query_norm mode
            # Calculate L2 norms for dynamic heads
            q_for_routing = q.permute(0, 2, 1, 3).reshape(-1, self.h, self.d_k)
            logits = torch.stack([
                torch.norm(q_for_routing[:, i, :], p=2, dim=1) 
                for i in range(self.h_shared, self.h)
            ], dim=1)  # [bs*seq_len, n_dynamic_heads]
            
            # Normalize logits
            logits_std = logits.std(dim=1, keepdim=True)
            logits_norm = logits / (logits_std / 1)
            gates = F.softmax(logits_norm, dim=1)  # [bs*seq_len, n_dynamic_heads]
        
        # Select top-k heads
        _, indices = torch.topk(gates, k=self.h_selected, dim=1)
        dynamic_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
        
        self.dynamic_scores = gates * dynamic_mask
        
        # Update routing statistics
        self.head_routing_probs = gates.mean(dim=0)
        self.head_selections = dynamic_mask.sum(dim=0)
        
        # Handle shared heads routing
        # All shared heads have equal weight of 1.0
        self.shared_scores = torch.ones(bs, seq_len, self.h_shared).to(q.device)
        
        dynamic_scores_reshaped = self.dynamic_scores.view(bs, seq_len, -1)
        routing_mask = torch.zeros(bs, seq_len, self.h).to(q.device)
        routing_mask[:, :, :self.h_shared] = 1.0  # Shared heads always active
        routing_mask[:, :, self.h_shared:] = dynamic_scores_reshaped  # Add dynamic head weights
        
        # Reshape routing mask to match attention dimensions [bs, h, seq_len, 1]
        routing_mask = routing_mask.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention
        attn = self.dropout(torch.softmax(scores, dim=-1))
        
        # Save attention maps for visualization
        self.attention_maps = attn.detach().clone()  # [bs, h, seq_len, seq_len]
        
        context = torch.matmul(attn, v)  # [bs, h, seq_len, d_k]
        
        # Apply routing mask
        context = context * routing_mask
        
        # Combine heads
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(context)



class RouterKT(nn.Module, DLSequentialKTModel):
    model_name = "RouterKT"

    def __init__(self, params, objects):
        super(RouterKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["RouterKT"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.encoder_layer = RouterKTArchitecture(params)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])
        self.separate_qa = model_config["separate_qa"]

    def base_emb(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        separate_qa = self.separate_qa
        num_concept = self.objects["dataset"]["q_table"].shape[1]

        # c_ct
        concept_emb = self.embed_layer.get_emb_fused1("concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        if separate_qa:
            interaction_seq = num_concept * batch["correctness_seq"].unsqueeze(-1)
            interaction_emb = self.embed_layer.get_emb_fused1(
                "interaction", q2c_transfer_table, q2c_mask_table, batch["question_seq"], other_item_index=interaction_seq)
        else:
            # e_{(c_t, r_t)} = c_{c_t} + r_{r_t}
            interaction_emb = self.embed_layer.get_emb("interaction", batch["correctness_seq"]) + concept_emb

        return concept_emb, interaction_emb

    def get_latent(self, batch):
        separate_qa = self.separate_qa
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        # c_{c_t}和e_(ct, rt)
        concept_emb, interaction_emb = self.base_emb(batch)
        # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        concept_variation_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        # mu_{q_t}
        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        # mu_{q_t} * d_ct + c_ct
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        # f_{(c_t, r_t)}中的r_t
        interaction_variation_emb = self.embed_layer.get_emb("interaction_var", batch["correctness_seq"])
        if separate_qa:
            # uq * f_(ct,rt) + e_(ct,rt)
            interaction_emb = interaction_emb + question_difficulty_emb * interaction_variation_emb
        else:
            # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            interaction_emb = \
                interaction_emb + question_difficulty_emb * (interaction_variation_emb + concept_variation_emb)

        encoder_input = {
            "question_emb": question_emb,
            "interaction_emb": interaction_emb,
            "question_difficulty_emb": question_difficulty_emb
        }

        latent = self.encoder_layer(encoder_input)
        return latent

    def forward(self, batch):
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        concept_emb, _ = self.base_emb(batch)
        concept_variation_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        question_emb = concept_emb + question_difficulty_emb * concept_variation_emb
        latent = self.get_latent(batch)
        predict_layer_input = torch.cat((latent, question_emb), dim=2)
        predict_score_batch = self.predict_layer(predict_layer_input).squeeze(dim=-1)
        return predict_score_batch

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation
        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch, seq_start)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])

        question_difficulty_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        rasch_loss = (question_difficulty_emb[mask_seq] ** 2.).sum()

        # Get balance loss from MoH attention layers
        balance_loss = self.encoder_layer.get_balance_loss()

        # Combine losses
        model_config = self.params["models_config"]["RouterKT"]
        loss = predict_loss + \
               rasch_loss * self.params["loss_config"]["rasch loss"] + \
               balance_loss * model_config["balance_loss_weight"]

        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "rasch loss": {
                    "value": rasch_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "balance loss": {
                    "value": float(balance_loss * model_config["balance_loss_weight"] * num_sample),
                    "num_sample": num_sample
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_knowledge_state(self, batch):
        pass


class RouterKTArchitecture(nn.Module):
    def __init__(self, params):
        super(RouterKTArchitecture, self).__init__()
        self.params = params
        model_config = self.params["models_config"]["RouterKT"]
        num_block = model_config["num_block"]

        # Create transformer blocks with MoH attention
        self.question_encoder = nn.ModuleList([
            RouterTransformerLayer(params) for _ in range(num_block * 2)
        ])
        self.knowledge_encoder = nn.ModuleList([
            RouterTransformerLayer(params) for _ in range(num_block)
        ])

    def get_balance_loss(self):
        """Get balance loss from all MoH attention layers."""
        balance_loss = 0.0
        for block in self.knowledge_encoder:
            balance_loss += block.attn.get_balance_loss()
        for block in self.question_encoder:
            balance_loss += block.attn.get_balance_loss()
        return balance_loss

    def forward(self, batch):
        x = batch["question_emb"]
        y = batch["interaction_emb"]
        question_difficulty_emb = batch["question_difficulty_emb"]
        diff = None
        response = None

        # Knowledge encoder
        for block in self.knowledge_encoder:
            # Process interaction embeddings
            y = block(y, y, y, mask_flag=True, diff=diff, response=response, apply_pos=False)

        # Question encoder with alternating self-attention and cross-attention
        flag_first = True
        for block in self.question_encoder:
            if flag_first:
                # Self-attention on question embeddings
                x = block(x, x, x, mask_flag=True, diff=diff, response=response, apply_pos=False)
                flag_first = False
            else:
                # Cross-attention between question and interaction
                x = block(x, x, y, mask_flag=False, diff=diff, response=response, apply_pos=True)
                flag_first = True

        return x


class RouterTransformerLayer(nn.Module):
    def __init__(self, params):
        super(RouterTransformerLayer, self).__init__()
        self.params = params

        model_config = self.params["models_config"]["RouterKT"]
        dim_model = model_config["dim_model"]
        dim_ff = model_config["dim_ff"]
        dropout = model_config["dropout"]
        num_head = model_config["num_head"]
        num_shared_heads = model_config["num_shared_heads"]
        num_selected_heads = model_config["num_selected_heads"]
        key_query_same = model_config["key_query_same"]
        seq_len = model_config["seq_len"]
        routing_mode = model_config["routing_mode"]

        # MoH attention layer
        self.attn = MoHAttention(
            d_model=dim_model,
            d_feature=dim_model // num_head,
            n_heads=num_head,
            n_shared_heads=num_shared_heads,
            n_selected_heads=num_selected_heads,
            dropout=dropout,
            kq_same=key_query_same,
            seq_len=seq_len,
            routing_mode=routing_mode
        )

        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_model)
        )

        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, values, mask_flag, diff=None, response=None, apply_pos=True):
        # Create proper attention mask
        seq_len = query.size(1)
        if mask_flag:
            # Can see current and past values (mask=1)
            nopeek_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        else:
            # Can only see past values (mask=0)
            nopeek_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).bool()

        src_mask = (~nopeek_mask).to(query.device)

        # Apply MoH attention
        attn_output = self.attn(query, key, values, src_mask)

        # First residual connection and layer norm
        x = self.layer_norm1(query + self.dropout1(attn_output))

        # Apply feed-forward network if needed
        if apply_pos:
            ffn_output = self.ffn(x)
            x = self.layer_norm2(x + self.dropout2(ffn_output))

        return x
