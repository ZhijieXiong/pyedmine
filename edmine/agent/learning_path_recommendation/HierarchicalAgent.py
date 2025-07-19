from abc import abstractmethod
from copy import deepcopy


class HierarchicalAgent:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
    
        self.concept_rec_history = []
        self.question_rec_history = []
        self.question_rec_result = []
        self.state_history = []
        self.initial_seq_len = 0
        self.history_data = None
        self.learning_goals = None
            
    def reset(self, learning_goals, user_data=None):
        if user_data is None:
            self.history_data = {
                "question_seq": [],
                "correctness_seq": [],
                "mask_seq": [],
                "seq_len": 0
            }
            self.initial_seq_len = 0
        else:
            seq_len = user_data["seq_len"]
            self.initial_seq_len = seq_len
            self.history_data = {
                "question_seq": user_data["question_seq"],
                "correctness_seq": user_data["correctness_seq"],
                "mask_seq": [1] * seq_len,
                "seq_len": seq_len
            }
        self.learning_goals = learning_goals
        self.concept_rec_history = []
        self.question_rec_history = []
        self.state_history = []
        
    def update(self, current_state=None, next_rec_result=None):
        if current_state is not None:
            self.state_history.append(current_state)
        if next_rec_result is not None:
            next_rec_que, next_que_correctness = next_rec_result
            self.history_data["question_seq"].append(next_rec_que)
            self.history_data["correctness_seq"].append(next_que_correctness)
            self.history_data["mask_seq"].append(1)
            self.history_data["seq_len"] += 1
            
    def output_learning_history(self):
        return {
            "learning_goals": deepcopy(self.learning_goals),
            "history_data": deepcopy(self.history_data),
            "state_history": deepcopy(self.state_history),
            "concept_rec_history": deepcopy(self.concept_rec_history),
            "question_rec_history": deepcopy(self.question_rec_history)
        }
        
    def render(self):
        learning_goal = self.learning_goals[0]
        state = self.state_history[0]
        master_th = self.params["evaluate_config"]["master_threshold"]
        
        # 首行日志
        initial_ks = float(state[learning_goal])
        if initial_ks < master_th:
            msg = f"learning goal: c{learning_goal:<4}, initial knowledge state of c{learning_goal}: {str(initial_ks)[:4]}"
            self.objects["logger"].info(msg)

        # 准备数据
        correctness_seq = self.history_data["correctness_seq"][self.initial_seq_len:]
        i = 0

        # 每个推荐 concept 的记录
        for rec_c, rec_qs in zip(self.concept_rec_history, self.question_rec_history):
            state = self.state_history[i]
            rec_c_state = float(state[rec_c])
            line = f"    learning c{rec_c} , initial knowledge state of c{rec_c}: {str(rec_c_state)[:4]}"
            practiced_qs = []
            current_goal_changes = []
            learning_goal_changes = []
            for q_id in rec_qs:
                correctness = correctness_seq[i]
                correctness_str = "r" if correctness else "w"
                i += 1
                state = self.state_history[i]
                goal_state = float(state[learning_goal])
                rec_c_state = float(state[rec_c])
                practiced_qs.append((q_id, correctness_str))
                learning_goal_changes.append(goal_state)
                current_goal_changes.append(rec_c_state)
            line += "\n        "
            for q_id, c_str in practiced_qs:
                line += f"q{q_id} ({c_str}) --> "
            line += f"end\n        c{rec_c}: "
            for c_state in current_goal_changes:
                line += f"{str(c_state)[:4]} --> "
            line += f"end\n        c{learning_goal}: "
            for c_state in learning_goal_changes:
                line += f"{str(c_state)[:4]} --> "
            line += "end"
            self.objects["logger"].info(line)

        # 最后一行状态
        if i > 0:
            state = self.state_history[-1]
            final_ks = str(float(state[learning_goal]))[:4]
            self.objects["logger"].info(f"    final knowledge state of c{learning_goal}: {final_ks}\n")
    
    def achieve_single_goal(self):
        state = self.state_history[-1]
        learning_goal = self.learning_goals[0]
        master_th = self.params["evaluate_config"]["master_threshold"]
        return float(state[learning_goal]) >= master_th
    
    @abstractmethod
    def cal_reward(self):
        pass
    
    @abstractmethod
    def judge_done(self):
        pass
    
    @abstractmethod
    def rec_question(self):
        pass
