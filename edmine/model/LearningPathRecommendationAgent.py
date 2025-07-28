from abc import abstractmethod


class LPRAgent:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
    
    @abstractmethod
    def judge_done(self, memory, master_th):
        pass
    
    @abstractmethod
    def recommend_qc(self, memory, master_th):
        pass
