import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class MIKT4LPR(nn.Module, DLSequentialKTModel):
    model_name = "MIKT4LPR"

    def __init__(self, params, objects):
        super(MIKT4LPR, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["MIKT4LPR"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        dim_emb = model_config["dim_emb"]
        dim_state = model_config["dim_state"]
        dropout = model_config["dropout"]
        seq_len = model_config["seq_len"]

        self.embed_concept = nn.Parameter(torch.rand(num_concept, dim_emb))
        self.embed_question = nn.Parameter(torch.rand(num_question, dim_emb))
        self.embed_question_diff = nn.Embedding(num_question, 1)
        self.embed_correctness = nn.Embedding(2, dim_emb)
        self.embed_position = nn.Parameter(torch.rand(seq_len, dim_emb))
        self.embed_time = nn.Parameter(torch.rand(seq_len, dim_state))
        self.embed_coarse_state = nn.Parameter(torch.rand(1, dim_state))
        self.embed_fine_state = nn.Parameter(torch.rand(num_concept, dim_state))
        nn.init.xavier_uniform_(self.embed_question)
        nn.init.xavier_uniform_(self.embed_concept)
        nn.init.xavier_uniform_(self.embed_position)
        
        self.pro_linear = nn.Linear(dim_emb, dim_emb)
        self.skill_linear = nn.Linear(dim_emb, dim_emb)
        self.pro_change = nn.Linear(dim_emb, dim_emb)
        self.obtain1_linear = nn.Linear(dim_emb, dim_emb)
        self.obtain2_linear = nn.Linear(dim_emb, dim_emb)
        self.all_obtain = nn.Linear(dim_emb, dim_emb)
        self.predict_attn = nn.Linear(3 * dim_emb, dim_emb)
        
        # Concept-level integration module (no auxiliary loss)
        # Projects fine-grained state to scalar mastery, then uses mastery as feature
        # ===== New: Concept-level readout for mastery probability =====
        self.concept_readout = nn.Sequential(
            nn.Linear(dim_state, dim_state // 2),
            nn.ReLU(),
            nn.Linear(dim_state // 2, 1)
        )
        # map scalar mastery to embedding
        self.question_mastery_proj = nn.Linear(1, dim_emb)
        
        self.all_forget = nn.Sequential(
            nn.Linear(2 * dim_state, dim_state),
            nn.ReLU(),
            nn.Linear(dim_state, dim_state),
            nn.Sigmoid()
        )
        
        self.now_obtain = nn.Sequential(
            nn.Linear(dim_emb, dim_state),
            nn.Tanh(),
            nn.Linear(dim_state, dim_state),
            nn.Tanh()
        )
        
        self.pro_ability = nn.Sequential(
            nn.Linear(3 * dim_emb + dim_emb, dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb, 1)
        )

        self.skill_forget = nn.Sequential(
            nn.Linear(3 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, dim_emb)
        )

        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, batch):
        model_config = self.params["models_config"]["MIKT4LPR"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        device = self.params["device"]

        last_question_seq = batch["question_seq"][:, :-1]
        next_question_seq = batch["question_seq"][:, 1:]
        next_correctness_seq = batch["correctness_seq"][:, 1:]
        Q_table = self.objects["dataset"]["q_table_tensor"].float()

        seq_len = last_question_seq.shape[1]
        batch_size = last_question_seq.shape[0]

        concept_mean_emb = torch.matmul(Q_table, self.embed_concept) / (torch.sum(Q_table, dim=-1, keepdims=True) + 1e-8)
        question_diff = torch.sigmoid(self.embed_question_diff(torch.arange(num_question).to(device)))

        q_pro = self.pro_linear(self.embed_question)
        q_skill = self.skill_linear(self.embed_concept)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, Q_table == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        skill_attn = torch.matmul(attn, self.embed_concept)
        # 公式（3）， question_diff * self.pro_change(skill_mean)是公式（2）
        question_emb = self.dropout(skill_attn + question_diff * self.pro_change(concept_mean_emb))
        time_gap_emb = F.embedding(torch.ones(batch_size).to(device).long(), self.embed_time)
        
        next_q_seq_emb = F.embedding(next_question_seq, question_emb)
        next_interaction_seq_emb = next_q_seq_emb + self.embed_correctness(next_correctness_seq)
        
        predict_socre = []
        # record per-step, per-concept mastery probabilities
        last_skill_time = torch.zeros((batch_size, num_concept)).to(device).long()  
        # conceptual knowledge state (fine-grained)
        fine_grained_state = self.embed_fine_state.unsqueeze(0).repeat(batch_size, 1, 1)  
        # domain knowledge state (coarse-grained)
        coarse_grained_state = self.embed_coarse_state.repeat(batch_size, 1)

        for t in range(seq_len):
            next_question = next_question_seq[:, t]
            next_q2c = F.embedding(next_question, Q_table).unsqueeze(1)
            next_question_emb = next_q_seq_emb[:, t]
            
            # ==== Concept-level readout before update =====
            # 通过一个MLP将每个知识点向量映射为一个概率值
            p_c_t = torch.sigmoid(self.concept_readout(fine_grained_state))
            mastery_emb = self.question_mastery_proj(torch.matmul(next_q2c, p_c_t).squeeze(-1))

            f1 = next_question_emb.unsqueeze(1)
            f2 = fine_grained_state

            concept_time_gap = t - next_q2c.squeeze(1) * last_skill_time
            concept_time_gap_emb = F.embedding(concept_time_gap.long(), self.embed_time)

            # 公式（4）
            current_coarse_state = coarse_grained_state * self.all_forget(
                self.dropout(torch.cat([coarse_grained_state, time_gap_emb], dim=-1))
            )

            # 公式（5）
            effect_all_state = current_coarse_state.unsqueeze(1).repeat(1, f2.shape[1], 1)
            concept_forget = torch.sigmoid(self.skill_forget(
                self.dropout(torch.cat([fine_grained_state, concept_time_gap_emb, effect_all_state], dim=-1)))
            )
            concept_forget = torch.masked_fill(concept_forget, next_q2c.transpose(-1, -2) == 0, 1)
            fine_grained_state = fine_grained_state * concept_forget

            now_pro_skill_attn = torch.matmul(f1, fine_grained_state.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, next_q2c == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
            
            now_need_state = torch.matmul(now_pro_skill_attn, fine_grained_state).squeeze(1)
            # 公式（7）的attn，其中now_pro_embed是Q_{q_t}，now_need_state是FHS_t，forget_now_all_state是\tilde{H_t}
            all_attn = torch.sigmoid(self.predict_attn(self.dropout(
                torch.cat([now_need_state, current_coarse_state, next_question_emb], dim=-1)
            )))
            # 公式（7）中的f_{q_t}
            now_need_state = torch.cat([(1 - all_attn) * now_need_state, all_attn * current_coarse_state], dim=-1)
            # 记录每个知识点上一次被练习的时刻
            last_skill_time = torch.masked_fill(last_skill_time, next_q2c.squeeze(1) == 1, t)
            # 公式（8），针对当前习题的能力
            now_ability = torch.sigmoid(self.pro_ability(torch.cat([now_need_state, next_question_emb, mastery_emb], dim=-1)))  # batch 1
            now_diff = F.embedding(next_question, question_diff)

            # 为了使MIKT有追踪学生知识状态的能力，对预测层进行修改
            now_output = torch.sigmoid(5 * (now_ability - now_diff))
            now_output = now_output.squeeze(-1)
            predict_socre.append(now_output)

            next_interaction_emb = next_interaction_seq_emb[:, t]
            # 公式（11）
            coarse_grained_state = current_coarse_state + torch.tanh(self.all_obtain(self.dropout(next_interaction_emb))).squeeze(1)
            # 公式（12）
            to_get = torch.tanh(self.now_obtain(self.dropout(next_interaction_emb))).unsqueeze(1)

            f1 = to_get
            f2 = fine_grained_state

            now_pro_skill_attn = torch.matmul(f1, f2.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, next_q2c == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
            # 公式（13）
            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)

            fine_grained_state = fine_grained_state + now_get

        return torch.vstack(predict_socre).T

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch,
        }

    def get_knowledge_state(self, batch):
        model_config = self.params["models_config"]["MIKT4LPR"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        device = self.params["device"]

        last_question_seq = batch["question_seq"][:, :-1]
        next_question_seq = batch["question_seq"][:, 1:]
        next_correctness_seq = batch["correctness_seq"][:, 1:]
        Q_table = self.objects["dataset"]["q_table_tensor"].float()

        seq_len = last_question_seq.shape[1]
        batch_size = last_question_seq.shape[0]

        concept_mean_emb = torch.matmul(Q_table, self.embed_concept) / (torch.sum(Q_table, dim=-1, keepdims=True) + 1e-8)
        question_diff = torch.sigmoid(self.embed_question_diff(torch.arange(num_question).to(device)))

        q_pro = self.pro_linear(self.embed_question)
        q_skill = self.skill_linear(self.embed_concept)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, Q_table == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        skill_attn = torch.matmul(attn, self.embed_concept)
        # 公式（3）， question_diff * self.pro_change(skill_mean)是公式（2）
        question_emb = self.dropout(skill_attn + question_diff * self.pro_change(concept_mean_emb))
        time_gap_emb = F.embedding(torch.ones(batch_size).to(device).long(), self.embed_time)
        
        next_q_seq_emb = F.embedding(next_question_seq, question_emb)
        next_interaction_seq_emb = next_q_seq_emb + self.embed_correctness(next_correctness_seq)
        
        predict_socre = []
        # record per-step, per-concept mastery probabilities
        concept_mastery_seq = []  
        last_skill_time = torch.zeros((batch_size, num_concept)).to(device).long()  
        # conceptual knowledge state (fine-grained)
        fine_grained_state = self.embed_fine_state.unsqueeze(0).repeat(batch_size, 1, 1)  
        # domain knowledge state (coarse-grained)
        coarse_grained_state = self.embed_coarse_state.repeat(batch_size, 1)

        for t in range(seq_len):
            next_question = next_question_seq[:, t]
            next_q2c = F.embedding(next_question, Q_table).unsqueeze(1)
            next_question_emb = next_q_seq_emb[:, t]
            
            # ==== Concept-level readout before update =====
            # 通过一个MLP将每个知识点向量映射为一个概率值
            p_c_t = torch.sigmoid(self.concept_readout(fine_grained_state))
            concept_mastery_seq.append(p_c_t.squeeze(-1))
            mastery_emb = self.question_mastery_proj(torch.matmul(next_q2c, p_c_t).squeeze(-1))

            f1 = next_question_emb.unsqueeze(1)
            f2 = fine_grained_state

            concept_time_gap = t - next_q2c.squeeze(1) * last_skill_time
            concept_time_gap_emb = F.embedding(concept_time_gap.long(), self.embed_time)

            # 公式（4）
            current_coarse_state = coarse_grained_state * self.all_forget(
                self.dropout(torch.cat([coarse_grained_state, time_gap_emb], dim=-1))
            )

            # 公式（5）
            effect_all_state = current_coarse_state.unsqueeze(1).repeat(1, f2.shape[1], 1)
            concept_forget = torch.sigmoid(self.skill_forget(
                self.dropout(torch.cat([fine_grained_state, concept_time_gap_emb, effect_all_state], dim=-1)))
            )
            concept_forget = torch.masked_fill(concept_forget, next_q2c.transpose(-1, -2) == 0, 1)
            fine_grained_state = fine_grained_state * concept_forget

            now_pro_skill_attn = torch.matmul(f1, fine_grained_state.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, next_q2c == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
            
            now_need_state = torch.matmul(now_pro_skill_attn, fine_grained_state).squeeze(1)
            # 公式（7）的attn，其中now_pro_embed是Q_{q_t}，now_need_state是FHS_t，forget_now_all_state是\tilde{H_t}
            all_attn = torch.sigmoid(self.predict_attn(self.dropout(
                torch.cat([now_need_state, current_coarse_state, next_question_emb], dim=-1)
            )))
            # 公式（7）中的f_{q_t}
            now_need_state = torch.cat([(1 - all_attn) * now_need_state, all_attn * current_coarse_state], dim=-1)
            # 记录每个知识点上一次被练习的时刻
            last_skill_time = torch.masked_fill(last_skill_time, next_q2c.squeeze(1) == 1, t)
            # 公式（8），针对当前习题的能力
            now_ability = torch.sigmoid(self.pro_ability(torch.cat([now_need_state, next_question_emb, mastery_emb], dim=-1)))  # batch 1
            now_diff = F.embedding(next_question, question_diff)

            # 为了使MIKT有追踪学生知识状态的能力，对预测层进行修改
            now_output = torch.sigmoid(5 * (now_ability - now_diff))
            now_output = now_output.squeeze(-1)
            predict_socre.append(now_output)

            next_interaction_emb = next_interaction_seq_emb[:, t]
            # 公式（11）
            coarse_grained_state = current_coarse_state + torch.tanh(self.all_obtain(self.dropout(next_interaction_emb))).squeeze(1)
            # 公式（12）
            to_get = torch.tanh(self.now_obtain(self.dropout(next_interaction_emb))).unsqueeze(1)

            f1 = to_get
            f2 = fine_grained_state

            now_pro_skill_attn = torch.matmul(f1, f2.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, next_q2c == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
            # 公式（13）
            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)

            fine_grained_state = fine_grained_state + now_get
            
        # stack predictions and mastery sequences
        concept_mastery_batch = torch.stack(concept_mastery_seq, dim=1)

        return concept_mastery_batch[torch.arange(batch_size), batch["seq_len"] - 2]
    