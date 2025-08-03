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


class A2C(RLBasedLPRAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        num_question, num_concept = self.objects["dataset"]["q_table"].shape
        action_model_config = self.params["models_config"]["action_model"]
        state_model_config = self.params["models_config"]["state_model"]
        
        self.c_actor = init_model(
            num_concept * 2,
            num_concept,
            action_model_config["num_layer"],
            softmax=True
        ).to(self.params["device"])
        self.c_critic = init_model(
            num_concept * 2,
            1,
            state_model_config["num_layer"],
            softmax=False
        ).to(self.params["device"])
        self.c_critic_delay = init_model(
            num_concept * 2,
            1,
            state_model_config["num_layer"],
            softmax=False
        ).to(self.params["device"])
        
        self.q_actor = init_model(
            num_concept * 2,
            num_question,
            action_model_config["num_layer"],
            softmax=True
        ).to(self.params["device"])
        self.q_critic = init_model(
            num_concept * 2,
            1,
            state_model_config["num_layer"],
            softmax=False
        ).to(self.params["device"])
        self.q_critic_delay = init_model(
            num_concept * 2,
            1,
            state_model_config["num_layer"],
            softmax=False
        ).to(self.params["device"])
        
        self.objects["lpr_models"] = {
            "concept_actor": self.c_actor,
            "concept_critic": self.c_critic,
            "question_actor": self.q_actor,
            "question_critic": self.q_critic
        }
        
    def eval(self):
        self.c_actor.eval()
        self.c_critic.eval()
        self.c_critic_delay.eval()
        self.q_actor.eval()
        self.q_critic.eval()
        self.q_critic_delay.eval()
        
    def train(self):
        self.q_actor.train()
        self.q_critic.train()
        self.c_actor.train()
        self.c_critic.train()
        
    def copy_state_model(self):
        self.c_critic_delay.load_state_dict(self.c_critic.state_dict())
        self.q_critic_delay.load_state_dict(self.q_critic.state_dict())
        self.c_critic_delay.eval()
        self.q_critic_delay.eval()
        
    def judge_done(self, memory, master_th=0.6):
        if memory.achieve_single_goal(master_th):
            return True
        max_question_attempt = self.params["agents_config"]["A2C"]["max_question_attempt"]
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
            c_id2rec = int(torch.argmax(self.c_actor(c_rec_model_input)))
        
        learning_goals = np.zeros(num_concept, dtype=int)
        learning_goals[c_id2rec] = 1
        q_rec_model_input = torch.cat(
            (knowledge_state, 
                torch.from_numpy(learning_goals).to(dtype=data_type).to(self.params["device"]))
        ).unsqueeze(dim=0)
        q_mask = q_table.T[c_id2rec].bool().unsqueeze(dim=0).to(self.params["device"])
        with torch.no_grad():
            q_values = self.q_actor(q_rec_model_input)
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
                    "reward": reward,
                    "over": float(i == (len(state_history) - 2))
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
            "over": []
        }
        for item in data:        
            learning_goals = item["learning_goals"]
            state = item["state"]
            next_state = item["next_state"]
            rec_concept = item["rec_concept"]
            rec_question = item["rec_question"]
            reward = item["reward"]
            over = item["over"]
            
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
            batch["over"].append(torch.tensor(over).long().to(self.params["device"]))
        for k, v in batch.items():
            batch[k] = torch.stack(v, dim=0)

        return batch
    
    def get_all_loss(self, batch):
        q_table = self.objects["dataset"]["q_table_tensor"]
        trainer_config = self.params["trainer_config"]
        gamma = trainer_config["gamma"]
        batch_size = batch["c_reward"].shape[0]
        
        c_value = self.c_critic(batch["cur_c_input"]).squeeze(dim=-1)
        with torch.no_grad():
            c_target = self.c_critic_delay(batch["next_c_input"]).squeeze(dim=-1)
        c_target = c_target * gamma * (1 - batch["over"]) + batch["c_reward"]
        c_state_loss = torch.nn.functional.mse_loss(c_value, c_target)
        
        c_advantage = (c_target - c_value).detach()
        c_prob = self.c_actor(batch["cur_c_input"])
        c_prob = c_prob.gather(dim=1, index=batch["cur_c"].unsqueeze(dim=-1)).squeeze(dim=-1)
        c_prob = (c_prob + 1e-8).log() * c_advantage
        c_action_loss = -c_prob.mean()
        
        q_mask = q_table.T[batch["cur_c"]].bool().to(self.params["device"])
        q_value = self.q_critic(batch["cur_q_input"]).squeeze(dim=-1)
        with torch.no_grad():
            q_target = self.q_critic_delay(batch["next_q_input"]).squeeze(dim=-1)
        q_target = q_target * gamma * (1 - batch["over"]) + batch["q_reward"]
        q_state_loss = torch.nn.functional.mse_loss(q_value, q_target)
        
        q_advantage = (q_target - q_value).detach()
        q_action_prob = self.q_actor(batch["cur_q_input"])
        # 非原地操作
        q_action_prob = q_action_prob.masked_fill(~q_mask, float('-inf'))
        q_action_prob = q_action_prob.gather(dim=1, index=batch["cur_q"].unsqueeze(dim=-1)).squeeze(dim=-1)
        q_action_prob = (q_action_prob + 1e-8).log() * q_advantage
        q_action_loss = -q_action_prob.mean()

        return {
            "concept_critic": {
                "total_loss": c_state_loss,
                "losses_value": {
                    "concept state loss": {
                        "value": c_state_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "concept_actor": {
                "total_loss": c_action_loss,
                "losses_value": {
                    "concept action loss": {
                        "value": c_action_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "question_critic": {
                "total_loss": q_state_loss,
                "losses_value": {
                    "question state loss": {
                        "value": q_state_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            },
            "question_actor": {
                "total_loss": q_action_loss,
                "losses_value": {
                    "question action loss": {
                        "value": q_action_loss.detach().cpu().item() * batch_size,
                        "num_sample": batch_size
                    }
                }
            }
        }
    