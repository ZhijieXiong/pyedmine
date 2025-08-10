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
        bes_setting = self.params["sequential_dlkt"]["bes_setting"]
        bes_use_time_decay = self.params["sequential_dlkt"]["bes_use_time_decay"]
        all_sample_path = self.params["all_sample_path"]
        save_all_sample = all_sample_path is not None

        predict_score_all = []
        ground_truth_all = []
        question_id_all = []
        question_all = []
        # result_all_batch是batch格式，即(num_batch * batch_size, seq_len)
        result_all_batch = []
        inference_result = {}
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
                "correctness_full": correctness_seq.detach().cpu().numpy(),
                "predict_score": predict_score_batch.detach().cpu().numpy(),
                "mask": batch["mask_seq"][:, 1:].detach().cpu().numpy()
            })

            # core指标计算
            question_all.append(torch.masked_select(question_seq[:, 1:], mask_seq[:, 1:]).detach().cpu().numpy())

        predict_score_all = np.concatenate(predict_score_all, axis=0)
        ground_truth_all = np.concatenate(ground_truth_all, axis=0)
        question_id_all = np.concatenate(question_id_all, axis=0)
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
            print("calculating user cold start metric ...")
            predict_score_cold_start_u = []
            ground_truth_cold_start_u = []
            question_id_in_user_cold_start = []
            for batch_result in result_all_batch:
                batch_size = batch_result["mask"].shape[0]
                seq_len = batch_result["mask"].shape[1]
                cold_start_mask = np.ones((batch_size, seq_len))
                cold_start_mask[:, user_cold_start:] = 0
                mask = np.logical_and(cold_start_mask, batch_result["mask"])
                predict_score_cold_start_u.append(batch_result["predict_score"][mask])
                ground_truth_cold_start_u.append(batch_result["correctness"][mask])
                question_id_in_user_cold_start.append(batch_result["question"][mask])
            predict_score_cold_start_u = np.concatenate(predict_score_cold_start_u, axis=0)
            ground_truth_cold_start_u = np.concatenate(ground_truth_cold_start_u, axis=0)
            inference_result["user_cold_start"] = get_kt_metric(ground_truth_cold_start_u, predict_score_cold_start_u)

            if question_cold_start >= 0:
                predict_score_cold_start_q_in_u = []
                ground_truth_cold_start_q_in_u = []
                cold_start_question = self.objects["cold_start_question"]
                question_id_in_user_cold_start = np.concatenate(question_id_in_user_cold_start, axis=0)
                print("calculating user cold start && question cold start metric ...")
                for q_id, ps, gt in zip(
                        question_id_in_user_cold_start, predict_score_cold_start_u, ground_truth_cold_start_u):
                    if q_id in cold_start_question:
                        predict_score_cold_start_q_in_u.append(ps)
                        ground_truth_cold_start_q_in_u.append(gt)
                inference_result["double_cold_start"] = get_kt_metric(ground_truth_cold_start_q_in_u,
                                                                      predict_score_cold_start_q_in_u)
        if question_cold_start >= 0:
            predict_score_cold_start_q = []
            ground_truth_cold_start_q = []
            cold_start_question = self.objects["cold_start_question"]
            print("calculating question cold start metric ...")
            for q_id, ps, gt in zip(question_id_all, predict_score_all, ground_truth_all):
                if q_id in cold_start_question:
                    predict_score_cold_start_q.append(ps)
                    ground_truth_cold_start_q.append(gt)
            inference_result["question_cold_start"] = get_kt_metric(ground_truth_cold_start_q, predict_score_cold_start_q)
            
        if que_start > 0:
            predict_score_warm_start_q = []
            ground_truth_warm_start_q = []
            warm_start_question = self.objects["warm_start_question"]
            print("calculating question warm start metric ...")
            for q_id, ps, gt in zip(question_id_all, predict_score_all, ground_truth_all):
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

        # 只在overall上计算，根据seq_start选择测试样本
        if bes_setting > 0:
            print("calculating BES metric ...")
            data4bes = []
            for batch_result in result_all_batch:
                for seq_len, question_seq, correctness_seq, predict_score_seq in zip(
                    batch_result["seq_len"], batch_result["question"], batch_result["correctness_full"], batch_result["predict_score"]
                ):
                    data4bes.append({
                        "question_seq": question_seq[:seq_len-1].tolist(),
                        "correctness_seq": correctness_seq[:seq_len].tolist(),
                        "predict_score_seq": predict_score_seq[:seq_len-1].tolist()
                    })
            # r表示样本参照
            all_concept_r = []
            all_question_r = []
            all_user_r = []
            all_predict_score = []
            all_correctness = []
            for user_data in data4bes:
                for ps in user_data["predict_score_seq"][seq_start-2:]:
                    all_predict_score.append(ps)
                for correctness in user_data["correctness_seq"][seq_start-1:]:
                    all_correctness.append(correctness)
            if bes_setting in [1, 2, 4, 6]:
                # concept
                q2c = self.objects["dataset"]["q2c"]
                concept_acc = self.objects["concept_acc"]
                for user_data in data4bes:
                    for q_id in user_data["question_seq"][seq_start-2:]:
                        q_concept_acc = []
                        for c_id in q2c[q_id]:
                            q_concept_acc.append(concept_acc[c_id])
                        all_concept_r.append(float(sum(q_concept_acc) / len(q_concept_acc)))
            if bes_setting in [1, 3, 4, 7]:
                # question
                question_acc = self.objects["question_acc"]
                for user_data in data4bes:
                    for q_id in user_data["question_seq"][seq_start-2:]:
                        all_question_r.append(question_acc[q_id])
            if bes_setting in [1, 2, 3, 5]:
                # student
                for user_data in data4bes:
                    correctness_seq = user_data["correctness_seq"]
                    for i in range(seq_start-1, len(correctness_seq)):
                        user_acc = float((sum(correctness_seq[:i]) + 1) / (i + 2))
                        all_user_r.append(user_acc)

            num_sample = max(len(all_concept_r), len(all_question_r), len(all_user_r))
            num_r_type = int(len(all_concept_r) > 0) + int(len(all_question_r) > 0) + int(len(all_user_r) > 0)
            all_final_r = [0] * num_sample
            for i in range(num_sample):
                if len(all_concept_r) > 0:
                    all_final_r[i] += all_concept_r[i]
                if len(all_question_r) > 0:
                    all_final_r[i] += all_question_r[i]
                if len(all_user_r) > 0:
                    all_final_r[i] += all_user_r[i]
                all_final_r[i] /= num_r_type

            # 样本对齐度，越大越“靠近数据参照”
            all_ps_r_align = [1 - abs(ps - r) for ps, r in zip(all_predict_score, all_final_r)]
            # 模型预测和gt之间的误差度
            all_model_brier = [(ps - y) ** 2 for ps, y in zip(all_predict_score, all_correctness)]
            # 样本参照和gt之间的误差度
            all_ref_brier = [(r - y) ** 2 for r, y in zip(all_final_r, all_correctness)]
            # 两者之差表示“模型在该样本上比数据参照更差多少”（差 >0 表示模型更差）
            all_delta_brier = [(mb - rb) for mb, rb in zip(all_model_brier, all_ref_brier)]
            # 样本偏差证据：当ps与r接近（align 高）且model_brier大于ref_brier（ΔBrier>0）时，
            # 偏差证据为正且较大——这是“被数据偏差误导并造成错误”的证据，若模型比参照好（ΔBrier<0），则样本偏差证据为负，说明模型抵抗了偏差
            bias_evidence = [ps_r_align * delta_brier for ps_r_align, delta_brier in zip(all_ps_r_align, all_delta_brier)]
            # 模型偏差分数曝光：BES
            BES = sum(bias_evidence) / len(bias_evidence)
            inference_result["BES"] = BES

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

        predict_score_all = []
        ground_truth_all = []
        for batch in tqdm(temp_loader, desc=f"last {'accumulative' if use_accumulative else 'non-accumulative'} multi step inference"):
            seq_len = int(batch["seq_len"][0])
            if seq_len <= multi_step:
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
        bes_setting = self.params["sequential_dlkt"]["bes_setting"]

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
                    performance = inference_result["double_cold_start"]
                    self.objects["logger"].info(
                        f"    double cold start performances (user_cold_start is {user_cold_start}, "
                        f"question_cold_start is {question_cold_start}) are AUC: "
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

            if bes_setting >= 1:
                BES = inference_result['BES']
                self.objects["logger"].info(
                    f"    model bias in{' s' if bes_setting in [1, 2, 3, 5] else ''}"
                    f"{' c' if bes_setting in [1, 2, 4, 6] else ''}"
                    f"{' q' if bes_setting in [1, 3, 4, 7] else ''} (seq_start is {seq_start}) are BES: {BES:<9.5}")
