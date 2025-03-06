import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from edmine.evaluator.DLEvaluator import DLEvaluator
from edmine.metric.knowledge_tracing import get_kt_metric, core_metric


class SequentialDLKTEvaluator(DLEvaluator):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def inference(self, model, data_loader):
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        cold_start = self.params["sequential_dlkt"]["cold_start"]
        multi_step = self.params["sequential_dlkt"]["multi_step"]

        predict_score_all = []
        ground_truth_all = []
        question_all = []
        # concept_all = []
        # result_all_batch是batch格式，即(num_batch * batch_size, seq_len)
        result_all_batch = []
        for batch in tqdm(data_loader, desc="one step inference"):
            correctness_seq = batch["correctness_seq"]
            mask_bool_seq = torch.ne(batch["mask_seq"], 0)
            predict_score = model.get_predict_score(batch, seq_start)["predict_score"].detach().cpu().numpy()
            ground_truth = torch.masked_select(correctness_seq[:, seq_start-1:], mask_bool_seq[:, seq_start-1:]).detach().cpu().numpy()
            predict_score_all.append(predict_score)
            ground_truth_all.append(ground_truth)

            # 冷启动计算
            question_seq = batch["question_seq"]
            predict_score_batch = model.get_predict_score(batch)["predict_score_batch"]
            result_all_batch.append({
                "question": question_seq[:, 1:].detach().cpu().numpy(),
                "label": correctness_seq[:, 1:].detach().cpu().numpy(),
                "predict_score": predict_score_batch.detach().cpu().numpy(),
                "mask": batch["mask_seq"][:, 1:].detach().cpu().numpy()
            })

            # core指标计算
            question_all.append(torch.masked_select(question_seq[:, 1:], mask_bool_seq[:, 1:]).detach().cpu().numpy())

        predict_score_all = np.concatenate(predict_score_all, axis=0)
        ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        inference_result = get_kt_metric(ground_truth_all, predict_score_all)

        inference_result["core"] = {
            "repeated": core_metric(predict_score_all, ground_truth_all, np.concatenate(question_all, axis=0), True),
            "non-repeated": core_metric(predict_score_all, ground_truth_all, np.concatenate(question_all, axis=0), False)
        }

        if cold_start >= 1:
            predict_label_cold_start = []
            ground_truth_cold_start = []
            for batch_result in result_all_batch:
                batch_size = batch_result["mask"].shape[0]
                seq_len = batch_result["mask"].shape[1]
                cold_start_mask = np.ones((batch_size, seq_len))
                cold_start_mask[:, cold_start:] = 0
                mask = np.logical_and(cold_start_mask, batch_result["mask"])
                predict_label_cold_start.append(batch_result["predict_score"][mask])
                ground_truth_cold_start.append(batch_result["label"][mask])
            predict_label_cold_start = np.concatenate(predict_label_cold_start, axis=0)
            ground_truth_cold_start = np.concatenate(ground_truth_cold_start, axis=0)
            inference_result["cold_start"] = get_kt_metric(ground_truth_cold_start, predict_label_cold_start)

        if multi_step > 1:
            inference_result["multi_step"] = {
                "non-accumulate": self.multi_step_inference(model, data_loader, False),
                "accumulate": self.multi_step_inference(model, data_loader, True)
            }

        return inference_result

    def multi_step_inference(self, model, data_loader, use_accumulative=True):
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        multi_step = self.params["sequential_dlkt"]["multi_step"]

        predict_score_all = []
        ground_truth_all = []
        for batch in tqdm(data_loader, desc=f"multi step inference, {'accumulative' if use_accumulative else 'non-accumulative'}"):
            seq_len = batch["correctness_seq"].shape[1]
            for i in range(seq_start - 1, seq_len - multi_step):
                if use_accumulative:
                    next_batch = deepcopy(batch)
                    for j in range(i, i + multi_step):
                        next_score = model.get_predict_score_at_target_time(next_batch, j)
                        mask = torch.ne(batch["mask_seq"][:, j], 0)
                        predict_score = torch.masked_select(next_score, mask).detach().cpu().numpy()
                        ground_truth_ = batch["correctness_seq"][:, j]
                        ground_truth = torch.masked_select(ground_truth_, mask).detach().cpu().numpy()
                        predict_score_all.append(predict_score)
                        ground_truth_all.append(ground_truth)
                        next_batch["correctness_seq"][:, i] = (next_score > 0.5).long()
                else:
                    target_question = batch["question_seq"][:, i:i + multi_step]
                    mask = torch.ne(batch["mask_seq"][:, i:i + multi_step], 0)
                    predict_score_ = model.get_predict_score_on_target_question(batch, i, target_question)
                    predict_score = torch.masked_select(predict_score_, mask).detach().cpu().numpy()
                    ground_truth_ = batch["correctness_seq"][:, i:i + multi_step]
                    ground_truth = torch.masked_select(ground_truth_, mask).detach().cpu().numpy()
                    predict_score_all.append(predict_score)
                    ground_truth_all.append(ground_truth)

        predict_score_all = np.concatenate(predict_score_all, axis=0)
        ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        return get_kt_metric(ground_truth_all, predict_score_all)

    def log_inference_results(self):
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        cold_start = self.params["sequential_dlkt"]["cold_start"]
        multi_step = self.params["sequential_dlkt"]["multi_step"]

        for data_loader_name, inference_result in self.inference_results.items():
            self.objects["logger"].info(f"evaluate result of {data_loader_name}")
            performance = inference_result
            self.objects["logger"].info(
                f"    overall performances (seq_start {seq_start}) are AUC: "
                f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            performance = inference_result["core"]["repeated"]
            self.objects["logger"].info(
                f"    core performances (seq_start {seq_start}, repeated) are AUC: "
                f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            performance = inference_result["core"]["non-repeated"]
            self.objects["logger"].info(
                f"    core performances (seq_start {seq_start}, non-repeated) are AUC: "
                f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            if cold_start >= 1:
                performance = inference_result["cold_start"]
                self.objects["logger"].info(
                    f"    cold start performances (cold_start is {cold_start}) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            if multi_step > 1:
                performance = inference_result['multi_step']["accumulate"]
                self.objects["logger"].info(
                    f"    multi step performances (seq_start {seq_start}, multi_step is {multi_step}, accumulative) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

                performance = inference_result['multi_step']["non-accumulate"]
                self.objects["logger"].info(
                    f"    multi step performances (seq_start {seq_start}, multi_step is {multi_step}, non-accumulative) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
