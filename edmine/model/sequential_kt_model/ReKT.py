import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class ReKT(nn.Module, DLSequentialKTModel):
    model_name = "ReKT"
    
    def __init__(self, params, objects):
        super(ReKT, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["ReKT"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        dim_emb = model_config["dim_emb"]
        dropout = model_config["dropout"]

        self.num_question = num_question
        self.num_concept = num_concept

        self.pro_embed = nn.Parameter(torch.rand(num_question, dim_emb))
        self.skill_embed = nn.Parameter(torch.rand(num_concept, dim_emb))

        self.ans_embed = nn.Parameter(torch.rand(2, dim_emb))

        self.out = nn.Sequential(
            nn.Linear(4 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_emb, 1)
        )

        self.dropout = nn.Dropout(p=dropout)

        self.time_embed = nn.Parameter(torch.rand(200, dim_emb))

        self.ls_state = nn.Parameter(torch.rand(1, dim_emb))
        self.c_state = nn.Parameter(torch.rand(1, dim_emb))

        self.pro_state = nn.Parameter(torch.rand(199, dim_emb))
        self.skill_state = nn.Parameter(torch.rand(199, dim_emb))

        self.obtain_pro_forget = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.Sigmoid()
        )
        self.obtain_pro_state = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb)
        )

        self.obtain_all_forget = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.Sigmoid()
        )

        self.obtain_skill_forget = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.Sigmoid()
        )
        self.obtain_skill_state = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb)
        )
        self.obtain_all_state = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb)
        )

        self.akt_pro_diff = nn.Parameter(torch.rand(num_question, 1))
        self.akt_pro_change = nn.Parameter(torch.rand(num_concept, dim_emb))

    def forward(self, batch):
        model_config = self.params["models_config"]["ReKT"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        
        last_problem = batch["question_seq"][:, :-1]
        # last_skill = batch["concept_seq"][:, :-1]
        # last_ans = batch["correctness_seq"][:, :-1]
        next_problem = batch["question_seq"][:, 1:]
        next_skill = batch["concept_seq"][:, 1:]
        next_ans = batch["correctness_seq"][:, 1:]
        
        device = self.params["device"]
        batch = last_problem.shape[0]
        seq = last_problem.shape[-1]

        next_pro_embed = F.embedding(next_problem, self.pro_embed) + \
            F.embedding(next_skill,self.skill_embed) + \
            F.embedding(next_problem, self.akt_pro_diff) * F.embedding(next_skill, self.akt_pro_change)

        next_X = next_pro_embed + F.embedding(next_ans.long(), self.ans_embed)

        last_pro_time = torch.zeros((batch, num_question)).to(device)  # batch pro
        last_skill_time = torch.zeros((batch, num_concept)).to(device)  # batch skill

        pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d
        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        all_state = self.ls_state.repeat(batch, 1)  # batch d

        last_pro_state = self.pro_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d
        last_skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)  # batch seq d

        batch_index = torch.arange(batch).to(device)

        all_time_gap = torch.ones((batch, seq)).to(device)
        all_time_gap_embed = F.embedding(all_time_gap.long(), self.time_embed)  # batch seq d

        res_p = []

        for now_step in range(seq):
            now_pro_embed = next_pro_embed[:, now_step]  # batch d

            now_item_pro = next_problem[:, now_step]  # batch
            now_item_skill = next_skill[:, now_step]

            last_batch_pro_time = last_pro_time[batch_index, now_item_pro]  # batch
            last_batch_pro_state = pro_state[batch_index, last_batch_pro_time.long()]  # batch d

            time_gap = now_step - last_batch_pro_time  # batch
            time_gap_embed = F.embedding(time_gap.long(), self.time_embed)  # batch d

            last_batch_skill_time = last_skill_time[batch_index, now_item_skill]  # batch
            last_batch_skill_state = skill_state[batch_index, last_batch_skill_time.long()]  # batch d

            skill_time_gap = now_step - last_batch_skill_time  # batch
            skill_time_gap_embed = F.embedding(skill_time_gap.long(), self.time_embed)  # batch d

            item_pro_state_forget = self.obtain_pro_forget(
                self.dropout(torch.cat([last_batch_pro_state, time_gap_embed], dim=-1)))
            last_batch_pro_state = last_batch_pro_state * item_pro_state_forget

            item_skill_state_forget = self.obtain_skill_forget(
                self.dropout(torch.cat([last_batch_skill_state, skill_time_gap_embed], dim=-1)))
            last_batch_skill_state = last_batch_skill_state * item_skill_state_forget

            item_all_state_forget = self.obtain_all_forget(
                self.dropout(torch.cat([all_state, all_time_gap_embed[:, now_step]], dim=-1)))
            last_batch_all_state = all_state * item_all_state_forget

            last_pro_state[:, now_step] = last_batch_pro_state
            last_skill_state[:, now_step] = last_batch_skill_state

            final_state = torch.cat(
                [last_batch_all_state, last_batch_pro_state, last_batch_skill_state, now_pro_embed], dim=-1)

            P = torch.sigmoid(self.out(self.dropout(final_state))).squeeze(-1)

            res_p.append(P)

            item_all_obtain = self.obtain_all_state(
                self.dropout(torch.cat([last_batch_all_state, next_X[:, now_step]], dim=-1)))
            item_all_state = last_batch_all_state + torch.tanh(item_all_obtain)

            all_state = item_all_state

            pro_get = next_X[:, now_step]
            skill_get = next_X[:, now_step]

            item_pro_obtain = self.obtain_pro_state(
                self.dropout(torch.cat([last_batch_pro_state, pro_get], dim=-1)))
            item_pro_state = last_batch_pro_state + torch.tanh(item_pro_obtain)

            item_skill_obtain = self.obtain_skill_state(
                self.dropout(torch.cat([last_batch_skill_state, skill_get], dim=-1)))
            item_skill_state = last_batch_skill_state + torch.tanh(item_skill_obtain)

            last_pro_time[batch_index, now_item_pro] = now_step
            pro_state[:, now_step] = item_pro_state

            last_skill_time[batch_index, now_item_skill] = now_step
            skill_state[:, now_step] = item_skill_state

        res_p = torch.vstack(res_p).T

        return res_p
    
    def get_predict_score(self, batch, seq_start=2):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_bool_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_knowledge_state(self, batch):
        pass
    