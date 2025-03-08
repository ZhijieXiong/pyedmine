import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from edmine.evaluator.DLEvaluator import DLEvaluator
from edmine.metric.knowledge_tracing import get_kt_metric


class DLCDEvaluator(DLEvaluator):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def inference(self, model, data_loader):
        predict_score_all = []
        predict_score_cold_start_u = []
        predict_score_cold_start_q = []
        ground_truth_all = []
        ground_truth_cold_start_u = []
        ground_truth_cold_start_q = []
        cold_start_user = self.objects["cold_start_user"]
        cold_start_question = self.objects["cold_start_question"]
        for batch in tqdm(data_loader, desc="evaluating ..."):
            question_id = batch["question_id"].detach().cpu().numpy()
            user_id = batch["user_id"].detach().cpu().numpy()
            predict_score = model.get_predict_score(batch)["predict_score"].detach().cpu().numpy()
            ground_truth = batch["correctness"].detach().cpu().numpy()
            predict_score_all.append(predict_score)
            ground_truth_all.append(ground_truth)
            ps_q = []
            ps_u = []
            gt_q = []
            gt_u = []
            for q_id, u_id, ps, gt in zip(question_id, user_id, predict_score, ground_truth):
                if q_id in cold_start_question:
                    ps_q.append(ps)
                    gt_q.append(gt)
                if u_id in cold_start_user:
                    ps_u.append(ps)
                    gt_u.append(gt)
            predict_score_cold_start_q += ps_q
            predict_score_cold_start_u += ps_u
            ground_truth_cold_start_q += gt_q
            ground_truth_cold_start_u += gt_u
            
        predict_score_all = np.concatenate(predict_score_all, axis=0)
        ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        inference_result = get_kt_metric(ground_truth_all, predict_score_all)
        inference_result["user_cold_start"] = get_kt_metric(ground_truth_cold_start_q, predict_score_cold_start_q)
        inference_result["question_cold_start"] = get_kt_metric(ground_truth_cold_start_u, predict_score_cold_start_u)
        

    def log_inference_results(self):
        user_cold_start = self.params["dlcd"]["user_cold_start"]
        question_cold_start = self.params["dlcd"]["question_cold_start"]

        for data_loader_name, inference_result in self.inference_results.items():
            self.objects["logger"].info(f"evaluate result of {data_loader_name}")
            performance = inference_result
            self.objects["logger"].info(
                f"    overall performances are AUC: "
                f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            if user_cold_start >= 1:
                performance = inference_result["user_cold_start"]
                self.objects["logger"].info(
                    f"    user cold start performances (user_cold_start is {user_cold_start}) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            if question_cold_start >= 1:
                performance = inference_result["question_cold_start"]
                self.objects["logger"].info(
                    f"    user cold start performances (question_cold_start is {question_cold_start}) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
                