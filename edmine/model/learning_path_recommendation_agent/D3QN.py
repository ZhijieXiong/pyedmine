import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

from edmine.model.learning_path_recommendation_agent.RLBasedLPRAgent import RLBasedLPRAgent


class DuelingDQN(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_actions, num_layer):
        super().__init__()
        # 特征提取部分（共享主干）
        layers = []
        for i in range(num_layer):
            if i == 0:
                layers.append(nn.Linear(dim_in, dim_hidden))
            else:
                layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers)

        # Value 分支：输出状态值 V(s)
        self.value_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)  # 输出单个值
        )

        # Advantage 分支：输出各个动作的优势 A(s, a)
        self.advantage_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_actions)  # 输出 num_actions 个值
        )

    def forward(self, x):
        features = self.feature_extractor(x)  # shape: (bs, dim_hidden)
        value = self.value_head(features)     # shape: (bs, 1)
        advantage = self.advantage_head(features)  # shape: (bs, num_actions)
        # Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values  # shape: (bs, num_actions)
    

class D3QN(RLBasedLPRAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        num_question, num_concept = self.objects["dataset"]["q_table"].shape
        c_rec_model_config = self.params["models_config"]["concept_rec_model"]
        q_rec_model_config = self.params["models_config"]["question_rec_model"]
        
        self.concept_rec_model = DuelingDQN(
            num_concept * 2,
            c_rec_model_config["dim_feature"],
            num_concept,
            c_rec_model_config["num_layer"]
        ).to(self.params["device"])
        self.question_rec_model = DuelingDQN(
            num_concept * 2,
            q_rec_model_config["dim_feature"],
            num_question,
            q_rec_model_config["num_layer"]
        ).to(self.params["device"])
        self.concept_rec_model_delay = DuelingDQN(
            num_concept * 2,
            c_rec_model_config["dim_feature"],
            num_concept,
            c_rec_model_config["num_layer"]
        ).to(self.params["device"])
        self.question_rec_model_delay = DuelingDQN(
            num_concept * 2,
            q_rec_model_config["dim_feature"],
            num_question,
            q_rec_model_config["num_layer"]
        ).to(self.params["device"])
        self.objects["lpr_models"] = {
            "concept_rec_model": self.concept_rec_model,
            "question_rec_model": self.question_rec_model
        }
        
    def eval(self):
        self.concept_rec_model.eval()
        self.concept_rec_model_delay.eval()
        self.question_rec_model.eval()
        self.question_rec_model_delay.eval()
        
    def train(self):
        self.concept_rec_model.train()
        self.question_rec_model.train()
        
    def prepare4next_epoch(self):
        self.concept_rec_model_delay.load_state_dict(self.concept_rec_model.state_dict())
        self.question_rec_model_delay.load_state_dict(self.question_rec_model.state_dict())
        self.concept_rec_model_delay.eval()
        self.question_rec_model_delay.eval()
    
    def judge_done(self, memory, master_th=0.6):
        if memory.achieve_single_goal(master_th):
            return True
        max_question_attempt = self.params["agents_config"]["D3QN"]["max_question_attempt"]
        num_question_his = 0
        for qs in memory.question_rec_history:
            num_question_his += len(qs)
        return num_question_his >= max_question_attempt
    
    def recommend_qc(self, memory, master_th=0.6, epsilon=0):
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        q_table = self.objects["dataset"]["q_table_tensor"]
        c2q = self.objects["dataset"]["c2q"]
        random_generator = self.objects["random_generator"]
        
        state = memory.state_history[-1]
        knowledge_state = state.to(self.params["device"])
        data_type = knowledge_state.dtype
        if random_generator.rand() < epsilon:
            eligible_concepts = [c_id for c_id in range(num_concept) if float(state[c_id]) < master_th]
            # 从未掌握的概念中随机选一个
            c_id2rec = random_generator.choice(eligible_concepts)
        else:
            learning_goals = np.zeros(num_concept, dtype=int)
            learning_goals[memory.learning_goals] = 1
            c_rec_model_input = torch.cat(
                (knowledge_state, 
                 torch.from_numpy(learning_goals).to(dtype=data_type).to(self.params["device"]))
            )
            with torch.no_grad():
                c_id2rec = int(torch.argmax(self.concept_rec_model(c_rec_model_input)))
        
        if random_generator.rand() < epsilon:
            q_id2rec = int(random_generator.choice(c2q[c_id2rec]))
        else:
            learning_goals = np.zeros(num_concept, dtype=int)
            learning_goals[c_id2rec] = 1
            q_rec_model_input = torch.cat(
                (knowledge_state, 
                 torch.from_numpy(learning_goals).to(dtype=data_type).to(self.params["device"]))
            )
            q_mask = q_table.T[c_id2rec].bool().to(self.params["device"])
            with torch.no_grad():
                q_values = self.question_rec_model(q_rec_model_input)
                q_values[~q_mask] = float('-inf')
                q_id2rec = int(torch.argmax(q_values))
        return c_id2rec, q_id2rec
            
    def done_data2rl_data(self, done_data):
        rl_data = []
        for item in done_data:
            state_history = item["state_history"]
            if len(state_history) <= 1:
                continue
            initial_state = state_history[0]
            last_state = state_history[-1]
            final_reward = 0
            for learning_goal in item["learning_goals"]:
                learning_gain = float(last_state[learning_goal]) - float(initial_state[learning_goal])
                final_reward += learning_gain / (1 - float(initial_state[learning_goal]))
            concept_rec_history = []
            question_rec_history = []
            for c, qs in zip(item["concept_rec_history"], item["question_rec_history"]):
                concept_rec_history += [c] * len(qs)
                question_rec_history += qs
            for i, (state, rec_c, rec_q) in enumerate(zip(state_history[:-1], concept_rec_history, question_rec_history)):
                if i == (len(state_history) - 2):
                    reward = final_reward
                else:
                    reward = 0
                rl_data.append({
                    "learning_goals": deepcopy(item["learning_goals"]),
                    "state": deepcopy(state),
                    "next_state": deepcopy(state_history[i+1]),
                    "rec_concept": rec_c,
                    "rec_question": rec_q,
                    "reward": reward
                })
        return rl_data
    
    def data2batch(self, data):
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        
        batch = {
            "cur_c_input": [],
            "next_c_input": [],
            "cur_q_input": [],
            "next_q_input": [],
            "cur_c": [],
            "cur_q": [],
            "c_reward": [],
            "q_reward": [],
        }
        for item in data:        
            learning_goals = item["learning_goals"]
            state = item["state"]
            next_state = item["next_state"]
            rec_concept = item["rec_concept"]
            rec_question = item["rec_question"]
            reward = item["reward"]
            
            cur_ks = state.to(self.params["device"])
            next_ks = next_state.to(self.params["device"])
            data_type = cur_ks.dtype
            final_goals = np.zeros(num_concept, dtype=int)
            final_goals[learning_goals] = 1
            cur_c_input = torch.cat(
                (cur_ks, 
                 torch.from_numpy(final_goals).to(dtype=data_type).to(self.params["device"]))
            )
            next_c_input = torch.cat(
                (next_ks, 
                 torch.from_numpy(final_goals).to(dtype=data_type).to(self.params["device"]))
            )

            current_goals = np.zeros(num_concept, dtype=int)
            current_goals[rec_concept] = 1
            cur_q_input = torch.cat(
                (cur_ks, 
                 torch.from_numpy(current_goals).to(dtype=data_type).to(self.params["device"]))
            )
            next_q_input = torch.cat(
                (next_ks, 
                 torch.from_numpy(current_goals).to(dtype=data_type).to(self.params["device"]))
            )
            
            batch["c_reward"].append(torch.tensor(reward).float().to(self.params["device"]))
            batch["q_reward"].append(torch.tensor(reward).float().to(self.params["device"]))
            batch["cur_c_input"].append(cur_c_input)
            batch["next_c_input"].append(next_c_input)
            batch["cur_q_input"].append(cur_q_input)
            batch["next_q_input"].append(next_q_input)
            batch["cur_c"].append(torch.tensor(rec_concept).long().to(self.params["device"]))
            batch["cur_q"].append(torch.tensor(rec_question).long().to(self.params["device"]))
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=0)

        return batch

    def get_all_loss(self, batch):
        q_table = self.objects["dataset"]["q_table_tensor"]
        trainer_config = self.params["trainer_config"]
        gamma = trainer_config["gamma"]
        batch_size = batch["c_reward"].shape[0]

        c_value = self.concept_rec_model(batch["cur_c_input"]).gather(dim=1, index=batch["cur_c"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # double DQN：用online网络选择动作，然后用target网络估计target state value
            next_c_action = self.concept_rec_model(batch["next_c_input"]).argmax(dim=1, keepdim=True)
            c_target = self.concept_rec_model_delay(batch["next_c_input"]).gather(1, next_c_action).squeeze(1)
        c_target = batch["c_reward"] + c_target * gamma
        concept_rec_loss = torch.nn.functional.mse_loss(c_value, c_target)

        q_mask = q_table.T[next_c_action].squeeze(dim=1).bool().to(self.params["device"])
        q_value = self.question_rec_model(batch["cur_q_input"]).gather(dim=1, index=batch["cur_q"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # double DQN
            next_q_values = self.question_rec_model(batch["next_q_input"])
            next_q_values[~q_mask] = float('-inf')
            next_q_action = next_q_values.argmax(dim=1, keepdim=True)
            q_target = self.question_rec_model_delay(batch["next_q_input"]).gather(1, next_q_action).squeeze(1)
        q_target = batch["q_reward"] + q_target * gamma
        question_rec_loss = torch.nn.functional.mse_loss(q_value, q_target)

        return {
            "concept_rec_model": {
                "total_loss": concept_rec_loss,
                "losses_value": {
                    "concept mse loss": {
                        "value": concept_rec_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "question_rec_model": {
                "total_loss": question_rec_loss,
                "losses_value": {
                    "question mse loss": {
                        "value": question_rec_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            }
        }
    