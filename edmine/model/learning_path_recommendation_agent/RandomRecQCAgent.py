from edmine.model.learning_path_recommendation_agent.RLBasedLPRAgent import RLBasedLPRAgent
    

class RandomRecQCAgent(RLBasedLPRAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
    
    def judge_done(self, memory, master_th=0.6):
        if memory.achieve_single_goal(master_th):
            return True
        evaluator_config = self.params["evaluator_config"]
        agent_name = evaluator_config["agent_name"]
        max_question_attempt = int(agent_name.split("-")[1])
        num_question_his = 0
        for qs in memory.question_rec_history:
            num_question_his += len(qs)
        return num_question_his >= max_question_attempt
    
    def recommend_qc(self, memory, master_th=0.6, epsilon=0):
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        evaluator_config = self.params["evaluator_config"]
        master_th = evaluator_config["master_threshold"]
        c2q = self.objects["dataset"]["c2q"]
        random_generator = self.objects["random_generator"]
        
        state = memory.state_history[-1]
        eligible_concepts = [c_id for c_id in range(num_concept) if float(state[c_id]) < master_th]
        c_id2rec = random_generator.choice(eligible_concepts)
        q_id2rec = int(random_generator.choice(c2q[c_id2rec]))
        return c_id2rec, q_id2rec
            