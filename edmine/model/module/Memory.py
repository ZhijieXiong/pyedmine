from copy import deepcopy


class LPRMemory:
    def __init__(self):    
        self.concept_rec_history = []
        self.question_rec_history = []
        self.state_history = []
        self.initial_seq_len = 0
        self.history_data = None
        self.learning_goals = None
        
    def reset(self, learning_goals, seq_data_keys=None, id_data_keys=None, user_data=None):
        if seq_data_keys is None:
            seq_data_keys = ["question_seq", "correctness_seq", "mask_seq"]
        if id_data_keys is None:
            id_data_keys = ["seq_len"]
        self.history_data = {seq_data_key: [] for seq_data_key in seq_data_keys}
        self.history_data["seq_len"] = 0
        if user_data is not None:
            seq_len = user_data["seq_len"]
            self.initial_seq_len = seq_len
            for id_data_key in id_data_keys:
                self.history_data[id_data_key] = user_data[id_data_key]
            for seq_data_key in seq_data_keys:
                if seq_data_key == "mask_seq":
                    self.history_data["mask_seq"] = [1] * seq_len
                else:
                    self.history_data[seq_data_key].extend(user_data[seq_data_key])
        self.learning_goals = learning_goals
        self.concept_rec_history = []
        self.question_rec_history = []
        self.state_history = []
        
    def update_rec_data(self, rec_concept, rec_question):
        if (len(self.concept_rec_history) == 0) or (rec_concept != self.concept_rec_history[-1]):
            self.concept_rec_history.append(rec_concept)
            self.question_rec_history.append([rec_question])
        else:
            self.question_rec_history[-1].append(rec_question)

    def update_history_data(self, current_state=None, next_rec_result=None):
        if current_state is not None:
            self.state_history.append(current_state)
        if next_rec_result is not None:
            # 如果后面有用需要其它信息（例如时间信息）的KT模型，需要在这更改数据更新
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
        
    def render(self, master_th):
        learning_goal = self.learning_goals[0]
        state = self.state_history[0]
        
        # 首行日志
        initial_ks = float(state[learning_goal])
        if initial_ks < master_th:
            msg = f"learning goal: c{learning_goal:<4}, initial knowledge state of c{learning_goal}: {str(initial_ks)[:4]}"
            print(msg)

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
            print(line)

        # 最后一行状态
        if i > 0:
            state = self.state_history[-1]
            final_ks = str(float(state[learning_goal]))[:4]
            print(f"    final knowledge state of c{learning_goal}: {final_ks}\n")
    
    def achieve_single_goal(self, master_th):
        state = self.state_history[-1]
        learning_goal = self.learning_goals[0]
        return float(state[learning_goal]) >= master_th