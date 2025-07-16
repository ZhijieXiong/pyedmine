import numpy as np

from edmine.agent.learning_path_recommendation.HierarchicalAgent import HierarchicalAgent


class RandomSingleGoalAgent(HierarchicalAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        
    def cal_reward(self):
        return 0
    
    def judge_done(self):
        concept_rec_strategy = self.params["agents_config"]["RandomAgent"]["concept_rec_strategy"]
        if concept_rec_strategy.startswith("random-"):
            max_stage = int(concept_rec_strategy.split("-")[1])
            if len(self.concept_rec_history) >= max_stage:
                max_attempt_per_concept = self.params["agents_config"]["RandomAgent"]["max_attempt_per_concept"]
                cur_stage = len(self.concept_rec_history) - 1
                last_stage_qs = self.question_rec_history[cur_stage]
                if len(last_stage_qs) >= max_attempt_per_concept:
                    return True
        
        return self.achieve_goals()
        
    def rec_question(self):
        state = self.state_history[-1]
        agent_config = self.params["agents_config"]["RandomAgent"]
        random_generator = self.objects["random_generator"]
        q2c = self.objects["dataset"]["q2c"]
        num_concept = agent_config["num_concept"]
        
        if len(self.concept_rec_history) == 0:
            c_id2rec = random_generator.randint(0, num_concept)
            self.concept_rec_history.append(c_id2rec)
            self.question_rec_history.append([])
        else:
            master_th = self.params["evaluate_config"]["master_threshold"]
            max_attempt_per_concept = agent_config["max_attempt_per_concept"]
            last_stage_rec_c = self.concept_rec_history[-1]
            cur_stage = len(self.concept_rec_history) - 1
            last_stage_qs = self.question_rec_history[cur_stage]
            if (state[last_stage_rec_c] > master_th) or (len(last_stage_qs) >= max_attempt_per_concept):
                eligible_concepts = [c_id for c_id in range(num_concept) if state[c_id] < master_th]
                # 从未掌握的概念中随机选一个
                c_id2rec = random_generator.choice(eligible_concepts)
                self.concept_rec_history.append(c_id2rec)
                self.question_rec_history.append([])
            else:
                c_id2rec = last_stage_rec_c
        
        cur_stage = len(self.concept_rec_history) - 1
        q_id2rec = random_generator.choice(q2c[c_id2rec])
        self.question_rec_history[cur_stage].append(q_id2rec)
        
        return q_id2rec
