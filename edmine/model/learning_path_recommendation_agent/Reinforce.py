import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

from edmine.model.learning_path_recommendation_agent.RLBasedLPRAgent import RLBasedLPRAgent


def init_model(dim_in, dim_out, num_layer, softmax=True):
    layers = []
    for i in range(num_layer):
        if i == 0:
            layers.append(nn.Linear(dim_in, dim_out))
        else:
            layers.append(nn.Linear(dim_out, dim_out))
        layers.append(nn.ReLU())
    if softmax:
        layers.append(nn.Linear(dim_out, dim_out))
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)
    

class Reinforce(RLBasedLPRAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        num_question, num_concept = self.objects["dataset"]["q_table"].shape
        action_model_config = self.params["models_config"]["action_model"]
        state_model_config = self.params["models_config"]["state_model"]
        
        self.concept_action_model = init_model(
            num_concept * 2,
            num_concept,
            action_model_config["num_layer"],
            softmax=True
        ).to(self.params["device"])
        self.concept_state_model = init_model(
            num_concept * 2,
            1,
            state_model_config["num_layer"],
            softmax=False
        ).to(self.params["device"])
        
        self.question_action_model = init_model(
            num_concept * 2,
            num_question,
            action_model_config["num_layer"],
            softmax=True
        ).to(self.params["device"])
        self.question_state_model = init_model(
            num_concept * 2,
            1,
            state_model_config["num_layer"],
            softmax=False
        ).to(self.params["device"])
        
        self.objects["lpr_models"] = {
            "concept_action_model": self.concept_action_model,
            "concept_state_model": self.concept_state_model,
            "question_action_model": self.question_action_model,
            "question_state_model": self.question_state_model
        }
        
    def eval(self):
        self.concept_action_model.eval()
        self.concept_state_model.eval()
        self.question_action_model.eval()
        self.question_action_model.eval()
        
    def train(self):
        self.concept_action_model.train()
        self.concept_state_model.train()
        self.question_action_model.train()
        self.question_action_model.train()
    
    def judge_done(self, memory, master_th=0.6):
        if memory.achieve_single_goal(master_th):
            return True
        max_question_attempt = self.params["agents_config"]["Reinforce"]["max_question_attempt"]
        num_question_his = 0
        for qs in memory.question_rec_history:
            num_question_his += len(qs)
        return num_question_his >= max_question_attempt
    
    def recommend_qc(self, memory, master_th=0.6, epsilon=0):
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        q_table = self.objects["dataset"]["q_table_tensor"]
        
        state = memory.state_history[-1]
        knowledge_state = state.to(self.params["device"])
        data_type = knowledge_state.dtype
        learning_goals = np.zeros(num_concept, dtype=int)
        learning_goals[memory.learning_goals] = 1
        c_rec_model_input = torch.cat(
            (knowledge_state, 
                torch.from_numpy(learning_goals).to(dtype=data_type).to(self.params["device"]))
        ).unsqueeze(dim=0)
        with torch.no_grad():
            c_id2rec = int(torch.argmax(self.concept_action_model(c_rec_model_input)))
        
        learning_goals = np.zeros(num_concept, dtype=int)
        learning_goals[c_id2rec] = 1
        q_rec_model_input = torch.cat(
            (knowledge_state, 
                torch.from_numpy(learning_goals).to(dtype=data_type).to(self.params["device"]))
        ).unsqueeze(dim=0)
        q_mask = q_table.T[c_id2rec].bool().unsqueeze(dim=0).to(self.params["device"])
        with torch.no_grad():
            q_values = self.question_action_model(q_rec_model_input)
            q_values[~q_mask] = float('-inf')
            q_id2rec = int(torch.argmax(q_values))
        return c_id2rec, q_id2rec
            
    def done_data2rl_data(self, done_data):
        gamma = self.params["trainer_config"]["gamma"]
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
            num_step = len(concept_rec_history)
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
                    "reward": reward,
                    "return_value": final_reward * (gamma ** (num_step - i - 1)),
                    "remain_step": num_step - i - 1
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
            "c_return": [],
            "q_reward": [],
            "q_return": [],
            "remain_step": []
        }
        for item in data:        
            learning_goals = item["learning_goals"]
            state = item["state"]
            next_state = item["next_state"]
            rec_concept = item["rec_concept"]
            rec_question = item["rec_question"]
            reward = item["reward"]
            return_value = item["return_value"]
            remain_step = item["remain_step"]
            
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
            batch["c_return"].append(torch.tensor(return_value).float().to(self.params["device"]))
            batch["q_return"].append(torch.tensor(return_value).float().to(self.params["device"]))
            batch["cur_c_input"].append(cur_c_input)
            batch["next_c_input"].append(next_c_input)
            batch["cur_q_input"].append(cur_q_input)
            batch["next_q_input"].append(next_q_input)
            batch["cur_c"].append(torch.tensor(rec_concept).long().to(self.params["device"]))
            batch["cur_q"].append(torch.tensor(rec_question).long().to(self.params["device"]))
            batch["remain_step"].append(torch.tensor(remain_step).long().to(self.params["device"]))
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=0)

        return batch

    def get_all_loss(self, batch):
        q_table = self.objects["dataset"]["q_table_tensor"]
        trainer_config = self.params["trainer_config"]
        gamma = trainer_config["gamma"]
        batch_size = batch["c_reward"].shape[0]

        c_state_baseline = self.concept_state_model(batch["cur_c_input"]).squeeze(dim=-1)
        c_state_loss = nn.functional.mse_loss(c_state_baseline, batch["c_return"])
        
        c_advantage = batch["c_return"] - c_state_baseline.detach()
        c_action_prob = self.concept_action_model(batch["cur_c_input"]).gather(dim=1, index=batch["cur_c"].unsqueeze(dim=-1)).squeeze(dim=-1)
        c_action_prob = (c_action_prob + 1e-8).log() * c_advantage
        c_action_prob = c_action_prob * (gamma ** batch["remain_step"])
        c_action_loss = -c_action_prob.mean()
        
        q_mask = q_table.T[batch["cur_c"]].bool().to(self.params["device"])
        q_state_baseline = self.question_state_model(batch["cur_q_input"]).squeeze(dim=-1)
        q_state_loss = nn.functional.mse_loss(q_state_baseline, batch["q_return"])
        
        q_advantage = batch["q_return"] - q_state_baseline.detach()
        q_action_prob = self.question_action_model(batch["cur_q_input"])
        # 非原地操作
        q_action_prob = q_action_prob.masked_fill(~q_mask, float('-inf'))
        q_action_prob = q_action_prob.gather(dim=1, index=batch["cur_q"].unsqueeze(dim=-1)).squeeze(dim=-1)
        q_action_prob = (q_action_prob + 1e-8).log() * q_advantage
        q_action_prob = q_action_prob * (gamma ** batch["remain_step"])
        q_action_loss = -q_action_prob.mean()

        return {
            "concept_state_model": {
                "total_loss": c_state_loss,
                "losses_value": {
                    "concept state loss": {
                        "value": c_state_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "concept_action_model": {
                "total_loss": c_action_loss,
                "losses_value": {
                    "concept action loss": {
                        "value": c_action_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "question_state_model": {
                "total_loss": q_state_loss,
                "losses_value": {
                    "question state loss": {
                        "value": q_state_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "question_action_model": {
                "total_loss": q_action_loss,
                "losses_value": {
                    "question action loss": {
                        "value": q_action_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            }
        }
    