from abc import ABC, abstractmethod


class CognitiveDiagnosisModel:
    @abstractmethod
    def get_knowledge_state(self, user_id):
        """
        Estimates the knowledge state of users based on their interaction data.
        Returns a matrix where each element represents a user's mastery level (a value between 0 and 1) for a specific concept.
        :param user_id: 
        :return: A matrix of shape (num_users, num_concepts) where:
            Each row corresponds to a user.
            Each column corresponds to a concept.
            Each element is a value between 0 and 1, representing the user's mastery level for that concept.
        """
        pass
