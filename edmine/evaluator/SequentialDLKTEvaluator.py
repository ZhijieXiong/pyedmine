import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader

from edmine.evaluator.DLEvaluator import DLEvaluator
from edmine.metric.knowledge_tracing import get_kt_metric, core_metric
from edmine.utils.data_io import write_kt_file


class SequentialDLKTEvaluator(DLEvaluator):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def inference(self, model, data_loader):
        evaluate_overall = self.params["sequential_dlkt"]["evaluate_overall"]
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        que_start = self.params["sequential_dlkt"]["que_start"]
        use_core = self.params["sequential_dlkt"]["use_core"]
        question_cold_start = self.params["sequential_dlkt"]["question_cold_start"]
        user_cold_start = self.params["sequential_dlkt"]["user_cold_start"]
        multi_step = self.params["sequential_dlkt"]["multi_step"]
        multi_step_accumulate = self.params["sequential_dlkt"]["multi_step_accumulate"]
        multi_step_overall = self.params["sequential_dlkt"]["multi_step_overall"]
        all_sample_path = self.params["all_sample_path"]
        save_all_sample = all_sample_path is not None

        predict_score_all = []
        ground_truth_all = []
        question_id_all = []
        question_all = []
        # result_all_batch是batch格式，即(num_batch * batch_size, seq_len)
        result_all_batch = []
        inference_result = {}
        if evaluate_overall or (question_cold_start >= 0) or (user_cold_start >= 1) or save_all_sample:
            for batch in tqdm(data_loader, desc="one step inference"):
                correctness_seq = batch["correctness_seq"]
                mask_seq = torch.ne(batch["mask_seq"], 0)
                question_seq = batch["question_seq"]
                predict_result = model.get_predict_score(batch, seq_start)
                predict_score = predict_result["predict_score"].detach().cpu().numpy()
                ground_truth = torch.masked_select(correctness_seq[:, seq_start-1:], mask_seq[:, seq_start-1:]).detach().cpu().numpy()
                question_id = torch.masked_select(question_seq[:, seq_start-1:], mask_seq[:, seq_start-1:]).detach().cpu().numpy()
                predict_score_all.append(predict_score)
                ground_truth_all.append(ground_truth)
                question_id_all.append(question_id)

                # 冷启动计算
                question_seq = batch["question_seq"]
                predict_score_batch = predict_result["predict_score_batch"]
                result_all_batch.append({
                    "user_id": batch["user_id"].detach().cpu().numpy(),
                    "seq_len": batch["seq_len"].detach().cpu().numpy(),
                    "question": question_seq[:, 1:].detach().cpu().numpy(),
                    "correctness": correctness_seq[:, 1:].detach().cpu().numpy(),
                    "predict_score": predict_score_batch.detach().cpu().numpy(),
                    "mask": batch["mask_seq"][:, 1:].detach().cpu().numpy()
                })

                # core指标计算
                question_all.append(torch.masked_select(question_seq[:, 1:], mask_seq[:, 1:]).detach().cpu().numpy())

            predict_score_all = np.concatenate(predict_score_all, axis=0)
            ground_truth_all = np.concatenate(ground_truth_all, axis=0)
            inference_result.update(get_kt_metric(ground_truth_all, predict_score_all))

        if save_all_sample:
            all_sample_result = []
            for batch in result_all_batch:
                for i, (user_id, seq_len) in enumerate(zip(batch["user_id"], batch["seq_len"])):
                    seq_len = int(seq_len)
                    user_result = {
                        "user_id": int(user_id),
                        "seq_len": seq_len,
                        "correctness_seq": [-1] + batch["correctness"][i][:seq_len-1].tolist(),
                        "predict_score_seq": [-1] + batch["predict_score"][i][:seq_len-1].tolist()
                    }
                    all_sample_result.append(user_result)
            write_kt_file(all_sample_result, all_sample_path)
        
        if use_core:
            inference_result["core"] = {
                "repeated": core_metric(predict_score_all, ground_truth_all, np.concatenate(question_all, axis=0), True),
                "non-repeated": core_metric(predict_score_all, ground_truth_all, np.concatenate(question_all, axis=0), False)
            }

        if user_cold_start >= 1:
            predict_score_cold_start_u = []
            ground_truth_cold_start_u = []
            for batch_result in result_all_batch:
                batch_size = batch_result["mask"].shape[0]
                seq_len = batch_result["mask"].shape[1]
                cold_start_mask = np.ones((batch_size, seq_len))
                cold_start_mask[:, user_cold_start:] = 0
                mask = np.logical_and(cold_start_mask, batch_result["mask"])
                predict_score_cold_start_u.append(batch_result["predict_score"][mask])
                ground_truth_cold_start_u.append(batch_result["correctness"][mask])
            predict_score_cold_start_u = np.concatenate(predict_score_cold_start_u, axis=0)
            ground_truth_cold_start_u = np.concatenate(ground_truth_cold_start_u, axis=0)
            inference_result["user_cold_start"] = get_kt_metric(ground_truth_cold_start_u, predict_score_cold_start_u)

        if question_cold_start >= 0:
            predict_score_cold_start_q = []
            ground_truth_cold_start_q = []
            cold_start_question = self.objects["cold_start_question"]
            print("calculating question cold start metric ...")
            for q_id, ps, gt in zip(question_id, predict_score, ground_truth):
                if q_id in cold_start_question:
                    predict_score_cold_start_q.append(ps)
                    ground_truth_cold_start_q.append(gt)
            inference_result["question_cold_start"] = get_kt_metric(ground_truth_cold_start_q, predict_score_cold_start_q)
            
        if que_start > 0:
            predict_score_warm_start_q = []
            ground_truth_warm_start_q = []
            warm_start_question = self.objects["warm_start_question"]
            print("calculating question warm start metric ...")
            for q_id, ps, gt in zip(question_id, predict_score, ground_truth):
                if q_id in warm_start_question:
                    predict_score_warm_start_q.append(ps)
                    ground_truth_warm_start_q.append(gt)
            inference_result["question_warm_start"] = get_kt_metric(ground_truth_warm_start_q, predict_score_warm_start_q)
        
        if multi_step > 1:
            inference_result["multi_step"] = {}
            non = "" if multi_step_accumulate else "non-"
            if multi_step_overall:
                inference_result["multi_step"][f"overall-{non}accumulate"] = self.multi_step_inference_overall(model, data_loader, multi_step_accumulate)
            else:
                inference_result["multi_step"][f"last-{non}accumulate"] = self.multi_step_inference_last(model, data_loader, multi_step_accumulate)

        return inference_result

    def multi_step_inference_overall(self, model, data_loader, use_accumulative=True):
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        multi_step = self.params["sequential_dlkt"]["multi_step"]
        num_batch = len(data_loader)
        temp_loader = DataLoader(dataset=data_loader.dataset, batch_size=1)
        seq_len = next(iter(temp_loader))["correctness_seq"].shape[1]

        predict_score_all = []
        ground_truth_all = []
        progress_bar = tqdm(total=num_batch * (seq_len - multi_step - seq_start + 1), 
                            desc=f"overall {'accumulative' if use_accumulative else 'non-accumulative'} multi step inference")
        for batch in data_loader:
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
                progress_bar.update(1)
            
        predict_score_all = np.concatenate(predict_score_all, axis=0)
        ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        return get_kt_metric(ground_truth_all, predict_score_all)
    
    def multi_step_inference_last(self, model, data_loader, use_accumulative=True):
        multi_step = self.params["sequential_dlkt"]["multi_step"]
        temp_loader = DataLoader(dataset=data_loader.dataset, batch_size=1)
        seq_len = next(iter(temp_loader))["correctness_seq"].shape[1]

        predict_score_all = []
        ground_truth_all = []
        for batch in tqdm(temp_loader, desc=f"last {'accumulative' if use_accumulative else 'non-accumulative'} multi step inference"):
            seq_len = int(batch["seq_len"][0])
            if seq_len <= (multi_step):
                continue
            i = seq_len - multi_step
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
        evaluate_overall = self.params["sequential_dlkt"]["evaluate_overall"]
        seq_start = self.params["sequential_dlkt"]["seq_start"]
        que_start = self.params["sequential_dlkt"]["que_start"]
        use_core = self.params["sequential_dlkt"]["use_core"]
        question_cold_start = self.params["sequential_dlkt"]["question_cold_start"]
        user_cold_start = self.params["sequential_dlkt"]["user_cold_start"]
        multi_step = self.params["sequential_dlkt"]["multi_step"]
        multi_step_accumulate = self.params["sequential_dlkt"]["multi_step_accumulate"]
        multi_step_overall = self.params["sequential_dlkt"]["multi_step_overall"]

        for data_loader_name, inference_result in self.inference_results.items():
            if evaluate_overall:
                self.objects["logger"].info(f"evaluate result of {data_loader_name}")
                performance = inference_result
                self.objects["logger"].info(
                    f"    overall performances (seq_start {seq_start}) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
            
                if que_start > 0:
                    performance = inference_result["question_warm_start"]
                    self.objects["logger"].info(
                        f"    overall performances (seq_start {seq_start}, que_start {que_start}) are AUC: "
                        f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                        f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            if use_core:
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

            if user_cold_start >= 1:
                performance = inference_result["user_cold_start"]
                self.objects["logger"].info(
                    f"    user cold start performances (cold_start is {user_cold_start}) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
                
            if question_cold_start >= 0:
                performance = inference_result["question_cold_start"]
                self.objects["logger"].info(
                    f"    question cold start performances (cold_start is {question_cold_start}) are AUC: "
                    f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                    f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")

            if multi_step > 1:
                if multi_step_overall and multi_step_accumulate:
                    performance = inference_result['multi_step']["overall-accumulate"]
                    self.objects["logger"].info(
                        f"    overall accumulative multi step performances (seq_start is {seq_start}, multi_step is {multi_step}) are AUC: "
                        f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                        f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
                elif multi_step_overall and (not multi_step_accumulate):
                    performance = inference_result['multi_step']["overall-non-accumulate"]
                    self.objects["logger"].info(
                        f"    overall non-accumulative multi step performances (seq_start is {seq_start}, multi_step is {multi_step}) are AUC: "
                        f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                        f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
                elif (not multi_step_overall) and (not multi_step_accumulate):
                    performance = inference_result['multi_step']["last-non-accumulate"]
                    self.objects["logger"].info(
                        f"    last non-accumulative multi step performances (seq_start is {seq_start}, multi_step is {multi_step}) are AUC: "
                        f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                        f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
                else:
                    performance = inference_result['multi_step']["last-accumulate"]
                    self.objects["logger"].info(
                        f"    last accumulative multi step performances (seq_start is {seq_start}, multi_step is {multi_step}) are AUC: "
                        f"{performance['AUC']:<9.5}, ACC: {performance['ACC']:<9.5}, "
                        f"RMSE: {performance['RMSE']:<9.5}, MAE: {performance['MAE']:<9.5}, ")
