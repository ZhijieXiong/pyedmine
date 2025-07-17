import torch

from edmine.metric.learning_path_recommendation import promotion_report


class LPREvaluator:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.inference_results = {dataset_name: {} for dataset_name in self.objects["datasets"].keys()}
        self.agents = []
        self.all_learning_history = []
        self.cur_data_idx = 0
        
    def add_agents(self, n):
        test_data = self.objects["datasets"]["test"]
        agent_class = self.objects["agent_class"]
        for _ in range(n):
            if self.cur_data_idx >= len(test_data):
                break
            user_data = test_data[self.cur_data_idx]
            agent = agent_class(self.params, self.objects)
            agent.reset([user_data["learning_goal"]], user_data)
            self.agents.append(agent)
            self.cur_data_idx += 1
            
    def remove_done_agents(self, batch_observation, batch_state):
        remain_idx = []
        for i, (agent, observation, state) in enumerate(zip(self.agents, batch_observation, batch_state)):
            agent.update(current_state=state)
            done = agent.judge_done()
            if done:
                self.all_learning_history.append(agent.output_learning_history())
                agent.render()
                self.agents[i] = None
            else:
                remain_idx.append(i)
        remain_idx = torch.tensor(remain_idx).long().to(self.params["device"])
        batch_observation = batch_observation[remain_idx]
        batch_state = batch_state[remain_idx]
        self.agents = list(filter(lambda x: x is not None, self.agents))
        
        return batch_observation, batch_state
        

    def evaluate(self):
        test_data = self.objects["datasets"]["test"]
        env = self.objects["env_simulator"]
        batch_size = self.params["datasets_config"]["test"]["batch_size"]
        
        self.add_agents(batch_size)
        env_input_data = {"history_data": [agent.history_data for agent in self.agents]}
        batch_observation, batch_state = env.step(env_input_data)
        batch_observation, batch_state = self.remove_done_agents(batch_observation, batch_state)
        
        while len(self.agents) == 0 and self.cur_data_idx < len(test_data):
            self.add_agents(batch_size)
            env_input_data = {"history_data": [agent.history_data for agent in self.agents]}
            batch_observation, batch_state = env.step(env_input_data)
            batch_observation, batch_state = self.remove_done_agents(batch_observation, batch_state)
        
        while len(self.agents) > 0:
            next_rec_data = []
            for agent, observation, state in zip(self.agents, batch_observation, batch_state):
                next_rec_data.append({
                    "question_seq": int(agent.rec_question()),
                    "correctness_seq": 0,
                    "mask_seq": 1    
                })
            
            env_input_data = {
                "history_data": [agent.history_data for agent in self.agents],
                "next_rec_data": next_rec_data
            }
            batch_observation, _ = env.step(env_input_data)
            for i, (agent, observation) in enumerate(zip(self.agents, batch_observation)):
                q_id = next_rec_data[i]["question_seq"]
                next_rec_result = (q_id, int(observation > 0.5))
                agent.update(next_rec_result=next_rec_result)
            env_input_data = {"history_data": [agent.history_data for agent in self.agents]}
            batch_observation, batch_state = env.step(env_input_data)
            batch_observation, batch_state = self.remove_done_agents(batch_observation, batch_state)
                
            self.add_agents(batch_size - len(self.agents))
            if len(self.agents) == 0:
                break
            
            env_input_data = {"history_data": [agent.history_data for agent in self.agents]}
            batch_observation, batch_state = env.step(env_input_data)
            self.remove_done_agents(batch_observation, batch_state)
            
            while len(self.agents) == 0 and self.cur_data_idx < len(test_data):
                self.add_agents(batch_size)
                env_input_data = {"history_data": [agent.history_data for agent in self.agents]}
                batch_observation, batch_state = env.step(env_input_data)
                self.remove_done_agents(batch_observation, batch_state)      
                
        self.log_inference_results()     

    def log_inference_results(self):
        samples = list(filter(lambda x: len(x["state_history"]) > 1, self.all_learning_history))            
        steps = [5, 10, 20, 50]
        steps.sort()
        data2evaluate = {
            step: {
                "intial_scores": [],
                "final_scores": [],
                "path_lens": [],
            }
            for step in steps
        }
        for sample in samples:
            learning_goal = sample["learning_goals"][0]
            states = list(map(lambda x: float(x[learning_goal]), sample["state_history"]))
            for step in steps:
                data2evaluate[step]["path_lens"].append(min(step, len(states)-1))
                data2evaluate[step]["intial_scores"].append(states[0])
                data2evaluate[step]["final_scores"].append(states[min(step, len(states)-1)])
                if step > len(states):
                    break
        for step in steps:
            intial_scores = data2evaluate[step]["intial_scores"]
            final_scores = data2evaluate[step]["final_scores"]
            path_lens = data2evaluate[step]["path_lens"]
            step_performance = promotion_report(intial_scores, final_scores, path_lens)
            performance_str = ""
            for metric_name, metric_value in step_performance.items():
                performance_str += f"{metric_name}: {metric_value:<9.5}, "
            self.objects["logger"].info(f"step {step} performances are {performance_str}")
