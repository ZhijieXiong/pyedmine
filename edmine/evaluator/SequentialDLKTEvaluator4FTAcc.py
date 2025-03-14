import numpy as np
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

from edmine.evaluator.DLEvaluator import DLEvaluator
from edmine.metric.knowledge_tracing import get_kt_metric


class SequentialDLKTEvaluator4FTAcc(DLEvaluator):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def inference(self, model, data_loader):
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        predict_score_all = []
        ground_truth_all = []
        for batch in tqdm(data_loader, desc="one step inference"):
            q2c = self.objects["dataset"]["q2c"]
            question_seqs = batch["question_seq"][:, 1:].detach().cpu().numpy()
            correctness_seqs = batch["correctness_seq"][:, 1:].detach().cpu().numpy()
            predict_lens = (batch["seq_len"] - 1).detach().cpu().numpy()
            predict_score_seqs = model.get_predict_score(batch)["predict_score_batch"].detach().cpu().numpy()
            for q_seq, c_seq, predict_len, ps_seq in zip(question_seqs, correctness_seqs, predict_lens, predict_score_seqs):
                if predict_len < (seq_start-1):
                    continue
                history_count = defaultdict(int)
                history_correct = defaultdict(int)
                for q_id, correctness, ps in zip(q_seq[seq_start-2:predict_len], c_seq[seq_start-2:predict_len], ps_seq[seq_start-2:predict_len]):
                    c_ids = q2c[q_id]
                    for c_id in c_ids:
                        num_exercised = history_count[c_id]
                        if num_exercised == 0:
                            predict_score_all.append(ps)
                            ground_truth_all.append(correctness)
                        history_count[c_id] += 1
                        history_correct[c_id] += correctness
        return get_kt_metric(ground_truth_all, predict_score_all)

    def log_inference_results(self):
        for data_loader_name, inference_result in self.inference_results.items():
            self.objects["logger"].info(f"evaluate result of {data_loader_name}")
            performance = inference_result
            self.objects["logger"].info(
                f"    first trans performances are AUC: "
                f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
