from edmine.evaluator.DLEvaluator import DLEvaluator
from edmine.metric.exercise_recommendation import *


class DLEREvaluator(DLEvaluator):
    def __init__(self, params, objects):
        super().__init__(params, objects)

    def inference(self, model, data_loader):
        top_ns = self.params["dler"]["top_ns"]
        q2c = self.objects["dataset"]["q2c"]

        # data_loader第3个数据是模型需要的，不同模型要的数据不一样
        user_history_data, user_mlkc, _ = data_loader
        if model.model_name == "KG4EX":
            top_ns_rec_questions = model.get_top_ns(data_loader, top_ns, self.params["dler"] ["kg4ex"]["batch_size"], True)
        else:
            top_ns_rec_questions = model.get_top_ns(data_loader, top_ns, True)

        inference_result = {top_n: {} for top_n in top_ns}
        users_mlkc = []
        users_concepts = []
        users_questions = []
        users_recommended_questions = {top_n: [] for top_n in top_ns}
        for user_id in top_ns_rec_questions:
            users_mlkc.append(user_mlkc[user_id])
            user_history_answer = user_history_data[user_id]
            valid_end_idx = user_history_answer["valid_end_idx"]
            seq_len = user_history_answer["seq_len"]
            users_concepts.append(
                get_history_correct_concepts(
                    user_history_answer["question_seq"][:seq_len], 
                    user_history_answer["correctness_seq"][:seq_len], q2c
                )
            )
            users_questions.append(
                get_future_incorrect_questions(
                    user_history_answer["question_seq"][valid_end_idx:seq_len], 
                    user_history_answer["correctness_seq"][valid_end_idx:seq_len]
                )
            )
            for top_n in top_ns:
                users_recommended_questions[top_n].append(top_ns_rec_questions[user_id][top_n])

        for top_n in top_ns:
            inference_result[top_n]["KG4EX_ACC"] = kg4ex_acc(users_mlkc, users_recommended_questions[top_n], q2c, 0.7)
            inference_result[top_n]["KG4EX_NOV"] = kg4ex_novelty(users_concepts, users_recommended_questions[top_n], q2c)
            inference_result[top_n]["OFFLINE_ACC"] = offline_acc(users_questions, users_recommended_questions[top_n])
            inference_result[top_n]["OFFLINE_NDCG"] = offline_ndcg(users_questions, users_recommended_questions[top_n])
            inference_result[top_n]["PERSONALIZATION_INDEX"] = personalization_index(users_recommended_questions[top_n])

        return inference_result
    
    def log_inference_results(self):
        top_ns = self.params["dler"]["top_ns"]
        for data_loader_name, inference_result in self.inference_results.items():
            self.objects["logger"].info(f"evaluate result of {data_loader_name}")
            performance = inference_result
            for top_n in top_ns:
                top_n_performance = performance[top_n]
                performance_str = ""
                for metric_name, metric_value in top_n_performance.items():
                    performance_str += f"{metric_name}: {metric_value:<9.5}, "
                print(f"    top {top_n} performances are {performance_str}")
