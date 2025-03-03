from abc import ABC, abstractmethod


class ExerciseRecommendationModel(ABC):
    @abstractmethod
    def get_top_ns(self, data, top_ns):
        """
        根据输入的数据data（user的历史数据），返回top n推荐习题
        :param data:
        :param top_ns:
        :return:
        """
        pass
