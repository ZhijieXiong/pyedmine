from abc import abstractmethod


class LPRAgent:
    def __init__(self, params, objects):
        self.params = params
        self.objects = objects
    
    @abstractmethod
    def judge_done(self, memory, master_th=0.6):
        pass
    
    @abstractmethod
    def recommend_qc(self, memory, master_th=0.6, epsilon=0):
        pass
    
    def eval(self):
        pass
    
    def train(self):
        pass
