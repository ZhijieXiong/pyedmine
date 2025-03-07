import torch
import numpy as np

from edmine.metric.knowledge_tracing import get_kt_metric
from edmine.trainer.EpochTrainer import SingleModelEpochTrainer


class SequentialDLKTTrainer(SingleModelEpochTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def evaluate_dataset(self, model, data_loader):
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                correctness_seq = batch["correctness_seq"]
                mask_bool_seq = torch.ne(batch["mask_seq"], 0)
                score_result = model.get_predict_score(batch)
                predict_score = score_result["predict_score"].detach().cpu().numpy()
                ground_truth = torch.masked_select(correctness_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)

            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            if model.model_name == "DKT_KG4EX":
                ground_truth_all = [1] * len(predict_score_all)

        return get_kt_metric(ground_truth_all, predict_score_all)
