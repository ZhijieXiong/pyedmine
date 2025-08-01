from abc import abstractmethod

from edmine.model.LearningPathRecommendationAgent import LPRAgent


class RLBasedLPRAgent(LPRAgent):
    def __init__(self, params, objects):
        super().__init__(params, objects)
        
    @abstractmethod
    def done_data2rl_data(self, done_data):
        """
        transform history data of done memory to rl data which is agent required
        """
        pass