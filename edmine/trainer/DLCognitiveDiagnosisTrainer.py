import torch
import numpy as np

from edmine.metric.knowledge_tracing import get_kt_metric
from edmine.trainer.EpochTrainer import SingleModelEpochTrainer


class DLCognitiveDiagnosisTrainer(SingleModelEpochTrainer):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def evaluate_dataset(self, model, data_loader):
        model.eval()
        with torch.no_grad():
            predict_score_all = []
            ground_truth_all = []
            for batch in data_loader:
                predict_score = model.get_predict_score(batch)["predict_score"].detach().cpu().numpy()
                ground_truth = batch["correctness"].detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)
            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        return get_kt_metric(ground_truth_all, predict_score_all)