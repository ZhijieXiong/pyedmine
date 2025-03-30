import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class MIKT(nn.Module, DLSequentialKTModel):
    model_name = "MIKT"

    def __init__(self, params, objects):
        super(MIKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["MIKT"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        dim_emb = model_config["dim_emb"]
        dim_state = model_config["dim_state"]
        dropout = model_config["dropout"]
        seq_len = model_config["seq_len"]

        self.pro_embed = nn.Parameter(torch.rand(num_question, dim_emb))
        nn.init.xavier_uniform_(self.pro_embed)
        self.skill_embed = nn.Parameter(torch.rand(num_concept, dim_emb))
        nn.init.xavier_uniform_(self.skill_embed)
        self.var = nn.Parameter(torch.rand(num_question, dim_emb))
        self.change = nn.Parameter(torch.rand(num_question, 1))
        self.pos_embed = nn.Parameter(torch.rand(seq_len, dim_emb))
        nn.init.xavier_uniform_(self.pos_embed)
        self.skill_state = nn.Parameter(torch.rand(num_concept, dim_state))
        self.time_state = nn.Parameter(torch.rand(seq_len, dim_state))
        self.all_state = nn.Parameter(torch.rand(1, dim_state))

        self.all_forget = nn.Sequential(
            nn.Linear(2 * dim_state, dim_state),
            nn.ReLU(),
            nn.Linear(dim_state, dim_state),
            nn.Sigmoid()
        )

        self.ans_embed = nn.Embedding(2, dim_emb)
        self.lstm = nn.LSTM(2 * dim_emb, dim_emb, batch_first=True)

        self.now_obtain = nn.Sequential(
            nn.Linear(dim_emb, dim_state),
            nn.Tanh(),
            nn.Linear(dim_state, dim_state),
            nn.Tanh()
        )

        self.pro_diff_embed = nn.Parameter(torch.rand(num_question, dim_emb))
        self.pro_diff = nn.Embedding(num_question, 1)

        self.pro_linear = nn.Linear(dim_emb, dim_emb)
        self.skill_linear = nn.Linear(dim_emb, dim_emb)
        self.pro_change = nn.Linear(dim_emb, dim_emb)

        self.pro_guess = nn.Embedding(num_question, 1)
        self.pro_divide = nn.Embedding(num_question, 1)

        self.skill_state = nn.Parameter(torch.rand(num_concept, dim_state))

        self.pro_ability = nn.Sequential(
            nn.Linear(3 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb, 1)
        )

        self.obtain1_linear = nn.Linear(dim_emb, dim_emb)
        self.obtain2_linear = nn.Linear(dim_emb, dim_emb)

        self.pro_diff_judge = nn.Linear(dim_emb, 1)

        self.all_obtain = nn.Linear(dim_emb, dim_emb)

        self.skill_forget = nn.Sequential(
            nn.Linear(3 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, dim_emb)
        )

        self.do_attn = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, 1)
        )

        self.predict_attn = nn.Linear(3 * dim_emb, dim_emb)

        self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, batch):
        model_config = self.params["models_config"]["MIKT"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]

        last_problem = batch["question_seq"][:, :-1]
        next_problem = batch["question_seq"][:, 1:]
        next_ans = batch["correctness_seq"][:, 1:]
        Q_table = self.objects["dataset"]["q_table_tensor"].float()

        device = self.params["device"]

        seq_len = last_problem.shape[1]
        batch_size = last_problem.shape[0]

        pro_embed = self.pro_embed
        skill_embed = self.skill_embed
        # pro d
        skill_mean = torch.matmul(Q_table, skill_embed) / (torch.sum(Q_table, dim=-1, keepdims=True) + 1e-8)

        pro_idx = torch.arange(num_question).to(device)
        pro_diff = torch.sigmoid(self.pro_diff(pro_idx))

        q_pro = self.pro_linear(pro_embed)
        q_skill = self.skill_linear(self.skill_embed)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, Q_table == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        skill_attn = torch.matmul(attn, skill_embed)
        # 公式（3），pro_diff * self.pro_change(skill_mean)是公式（2）
        now_embed = skill_attn + pro_diff * self.pro_change(skill_mean)
        # pro_embed就是公式（3）中的Q_{q_t}
        pro_embed = self.dropout(now_embed)

        next_pro_rasch = F.embedding(next_problem, pro_embed)

        next_X = next_pro_rasch + self.ans_embed(next_ans)

        last_all_time = torch.ones(batch_size).to(device).long()

        time_embed = self.time_state  # seq_len d
        all_gap_embed = F.embedding(last_all_time, time_embed)  # batch d

        res_p = []

        last_skill_time = torch.zeros((batch_size, num_concept)).to(device).long()  # batch skill
        # conceptual knowledge state (fine-grained)
        skill_state = self.skill_state.unsqueeze(0).repeat(batch_size, 1, 1)  # batch skill d
        # domain knowledge state (coarse-grained)
        all_state = self.all_state.repeat(batch_size, 1)  # batch d

        for now_step in range(seq_len):
            now_pro = next_problem[:, now_step]  # batch
            now_pro2skill = F.embedding(now_pro, Q_table).unsqueeze(1)  # batch 1 skill
            # 当前时刻要预测的习题emb
            now_pro_embed = next_pro_rasch[:, now_step]  # batch d

            f1 = now_pro_embed.unsqueeze(1)  # batch 1 d
            f2 = skill_state  # batch skill d

            skill_time_gap = now_step - now_pro2skill.squeeze(1) * last_skill_time  # batch skill
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), time_embed)  # batch skill d

            now_all_state = all_state  # batch d
            # 公式（4）
            forget_now_all_state = now_all_state * self.all_forget(
                self.dropout(torch.cat([now_all_state, all_gap_embed], dim=-1)))

            effect_all_state = forget_now_all_state.unsqueeze(1).repeat(1, f2.shape[1], 1)
            # 公式（5）
            skill_forget = torch.sigmoid(self.skill_forget(
                self.dropout(torch.cat([skill_state, skill_time_gap_embed, effect_all_state], dim=-1))))
            skill_forget = torch.masked_fill(skill_forget, now_pro2skill.transpose(-1, -2) == 0, 1)
            skill_state = skill_state * skill_forget

            now_pro_skill_attn = torch.matmul(f1, skill_state.transpose(-1, -2)) / f1.shape[-1]  # batch 1 skill

            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)

            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill

            now_need_state = torch.matmul(now_pro_skill_attn, skill_state).squeeze(1)  # batch d
            # 公式（7）的attn，其中now_pro_embed是Q_{q_t}，now_need_state是FHS_t，forget_now_all_state是\tilde{H_t}
            all_attn = torch.sigmoid(self.predict_attn(self.dropout(
                torch.cat([now_need_state, forget_now_all_state, now_pro_embed], dim=-1)
            )))
            # 公式（7）中的f_{q_t}
            now_need_state = torch.cat([(1 - all_attn) * now_need_state, all_attn * forget_now_all_state], dim=-1)
            # 记录每个知识点上一次被练习的时刻
            last_skill_time = torch.masked_fill(last_skill_time, now_pro2skill.squeeze(1) == 1, now_step)
            # 公式（8），针对当前习题的能力
            now_ability = torch.sigmoid(self.pro_ability(torch.cat([now_need_state, now_pro_embed], dim=-1)))  # batch 1
            now_diff = F.embedding(now_pro, pro_diff)  # batch 1

            now_output = torch.sigmoid(5 * (now_ability - now_diff))

            now_output = now_output.squeeze(-1)

            res_p.append(now_output)

            now_X = next_X[:, now_step]  # batch d
            # 公式（11）
            all_state = forget_now_all_state + torch.tanh(self.all_obtain(self.dropout(now_X))).squeeze(1)
            # 公式（12）
            to_get = torch.tanh(self.now_obtain(self.dropout(now_X))).unsqueeze(1)  # batch 1 d

            f1 = to_get  # batch 1 d
            f2 = skill_state  # batch skill d

            now_pro_skill_attn = torch.matmul(f1, f2.transpose(-1, -2)) / f1.shape[-1]  # batch 1 skill
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)  # batch 1 skill
            # 公式（13）
            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)

            skill_state = skill_state + now_get

        P = torch.vstack(res_p).T

        return P

    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass


