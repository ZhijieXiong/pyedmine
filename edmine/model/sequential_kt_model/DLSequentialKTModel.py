import torch
from abc import abstractmethod

from edmine.model.KnowledgeTracingModel import KnowledgeTracingModel
from edmine.model.loss import binary_cross_entropy


class DLSequentialKTModel(KnowledgeTracingModel):
    model_type = "DLSequentialKTModel"

    def get_predict_loss(self, batch, seq_start=2):
        """
        Computes the prediction loss for a batch of data using binary cross-entropy loss and returns the total loss, detailed loss values, and prediction scores.
        :param batch: A dictionary containing the keys like "question_seq", "correctness_seq", "mask_seq" and so on.
        :param seq_start: An integer specifying the starting index of the sequence for which to compute the loss. Default is 2.
        :return: A dictionary containing: total_loss, losses_value (A dictionary with detailed loss information), predict_score (The predicted scores for the batch), predict_score_batch (the predicted scores reshaped to match the batch structure)
        """
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch, seq_start)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": predict_loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    @abstractmethod
    def get_predict_score(self, batch, seq_start=2):
        """
        Computes predicted scores for a batch of data using the model's forward method. Applies a mask to filter out padding values and extract valid predictions. Returns the filtered predictions and the full batch of predictions.
        :param batch: A dictionary containing the keys like "question_seq", "correctness_seq", "mask_seq" and so on.
        :param seq_start: An integer specifying the starting index of the sequence for which to compute the loss. Default is 2.
        :return: A dictionary containing predict_score (A tensor of predicted scores for valid entries in the sequence, filtered using the mask) and predict_score_batch (A tensor of predicted scores for the entire batch)
        """
        pass

    @abstractmethod
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        """
        Predicts the probability of a user answering a specific target question correctly at a given time step.
        :param batch: A dictionary containing the keys like "question_seq", "correctness_seq", "mask_seq" and so on.
        :param target_index: An integer specifying the time step (index) for which to make the prediction.
        :param target_question: A tensor containing the IDs of the target questions for which predictions are to be made.
        :return:
        """
        pass

    def get_predict_score_at_target_time(self, batch, target_index):
        """
        Predicts the probability of a user answering the next exercise correctly at a specified time step.
        :param batch: A dictionary containing the keys like "question_seq", "correctness_seq", "mask_seq" and so on.
        :param target_index: An integer specifying the time step (index) for which to make the prediction.
        :return:
        """
        # self.get_predict_score(batch)["predict_score_batch"]输出的是从第2（自然数）个时刻开始的预测结果
        # target_index是从0开始的，所以要-1
        predict_score_batch = self.get_predict_score(batch)["predict_score_batch"]
        return predict_score_batch[:, target_index-1]

