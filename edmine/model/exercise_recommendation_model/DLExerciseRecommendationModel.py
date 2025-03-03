from abc import abstractmethod

from edmine.model.ExerciseRecommendationModel import ExerciseRecommendationModel


class DLExerciseRecommendationModel(ExerciseRecommendationModel):
    @abstractmethod
    def train_one_step(self, one_step_data):
        pass

