from edmine.metric.learning_path_recommendation import promotion_report
from edmine.model.module.Memory import LPRMemory


class LPREvaluator:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
        self.memories = []
        self.done_data = []
        self.cur_data_idx = 0
        
    def add_memories(self, n):
        test_data = self.objects["data"]["test"]
        for _ in range(n):
            if self.cur_data_idx >= len(test_data):
                break
            user_data = test_data[self.cur_data_idx]
            memory = LPRMemory()
            if "learning_goal" in user_data:
                memory.reset([user_data["learning_goal"]], user_data=user_data)
            else:
                memory.reset(user_data["learning_goals"], user_data=user_data)
            self.memories.append(memory)
            self.cur_data_idx += 1
            
    def remove_done_memories(self, batch_observation, batch_state):
        evaluator_config = self.params["evaluator_config"]
        render = evaluator_config["render"]
        master_th = evaluator_config["master_threshold"]
        agent_name = evaluator_config["agent_name"]
        agent = self.objects["agents"][agent_name]
        
        remain_indices = []
        for i, memory in enumerate(self.memories):
            done = agent.judge_done(memory, master_th)
            if done:
                self.done_data.append(memory.output_learning_history())
                if render:
                    memory.render(master_th)
                self.memories[i] = None
            else:
                remain_indices.append(i)
                
        self.memories = [self.memories[remain_idx] for remain_idx in remain_indices]
        batch_observation = batch_observation[remain_indices]
        batch_state = batch_state[remain_indices]
        
        return batch_observation, batch_state

    def evaluate(self):
        evaluator_config = self.params["evaluator_config"]
        batch_size = evaluator_config["batch_size"]
        agent_name = evaluator_config["agent_name"]
        test_data = self.objects["data"]["test"]
        env = self.objects["env_simulator"]
        agent = self.objects["agents"][agent_name]
        agent.eval()
        
        self.add_memories(batch_size)
        env_input_data = {"history_data": [memory.history_data for memory in self.memories]}
        batch_observation, batch_state = env.step(env_input_data)
        for memory, state in zip(self.memories, batch_state):
            memory.update_history_data(current_state=state)
        batch_observation, batch_state = self.remove_done_memories(batch_observation, batch_state)
        
        while len(self.memories) > 0:
            # 推荐习题
            next_rec_data = []
            for memory, observation, state in zip(self.memories, batch_observation, batch_state):
                rec_concept, rec_question = agent.recommend_qc(memory, 0)
                memory.update_rec_data(int(rec_concept), int(rec_question))
                next_rec_data.append({
                    "question_seq": int(rec_question),
                    "correctness_seq": 0,
                    "mask_seq": 1    
                })
            env_input_data = {
                "history_data": [memory.history_data for memory in self.memories],
                "next_rec_data": next_rec_data
            }
            batch_observation, batch_state = env.step(env_input_data)
            for i, (memory, observation, state) in enumerate(zip(self.memories, batch_observation, batch_state)):
                q_id = next_rec_data[i]["question_seq"]
                # 如果后面有用需要其它信息（例如时间信息）的KT模型，需要在这更改数据更新
                next_rec_result = (q_id, int(observation > 0.5))
                memory.update_history_data(current_state=state, next_rec_result=next_rec_result)
            batch_observation, batch_state = self.remove_done_memories(batch_observation, batch_state)

            # 添加KT数据用于模拟（要获取新加入数据的state）
            if (len(self.memories) < batch_size) and (self.cur_data_idx < len(test_data)):
                n = len(self.memories)
                self.add_memories(batch_size - n)
                env_input_data = {"history_data": [memory.history_data for memory in self.memories]}
                batch_observation, batch_state = env.step(env_input_data)
                for memory, state in zip(self.memories[n:], batch_state[n:]):
                    memory.update_history_data(current_state=state)
                batch_observation, batch_state = self.remove_done_memories(batch_observation, batch_state)    
                
        self.log_inference_results()     

    def log_inference_results(self):
        samples = list(filter(lambda x: len(x["state_history"]) > 1, self.done_data))            
        steps = [5, 10, 20]
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
                performance_str += f"{metric_name}: {metric_value:<9.5f}, "
            self.objects["logger"].info(f"step {step} performances are {performance_str}")
