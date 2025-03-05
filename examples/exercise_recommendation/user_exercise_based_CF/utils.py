from edmine.metric.exercise_recommendation import *


def delete_test_data(users_data):
    # 只能用训练集数据和验证集数据计算习题相似矩阵
    for user_data in users_data: 
        valid_end_idx = user_data["valid_end_idx"]
        user_data["seq_len"] = valid_end_idx
        for k, v in user_data.items():
            if type(v) is list:
                seq_len = len(v)
                user_data[k] = v[:valid_end_idx] + [0] * (seq_len - valid_end_idx)


def get_performance(used_metrics, top_ns, users_data, rec_result, q2c, mlkc=None, delta=0.7):
    performance = {x: {} for x in top_ns}
    for metric in used_metrics:
        for top_n in top_ns:
            rec_ques = rec_result[top_n]
            if metric == "PERSONALIZATION_INDEX":
                rec_ques_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                performance[top_n][metric] = personalization_index(rec_ques_)
            if metric == "KG4EX_ACC" and mlkc is not None:
                rec_ques_ = []
                mlkc_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    mlkc_.append(mlkc[user_id])
                performance[top_n][metric] = kg4ex_acc(mlkc_, rec_ques_, q2c, delta)
            if metric == "KG4EX_NOV":
                history_correct_cs = {}
                for item_data in users_data:
                    user_id = item_data["user_id"]
                    seq_len = item_data["seq_len"]
                    question_seq = item_data["question_seq"][:seq_len]
                    correctness_seq = item_data["correctness_seq"][:seq_len]
                    history_correct_cs[user_id] = get_history_correct_concepts(question_seq, correctness_seq, q2c)

                rec_ques_ = []
                correct_cs_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    correct_cs_.append(history_correct_cs[user_id])
                performance[top_n][metric] = kg4ex_novelty(correct_cs_, rec_ques_, q2c)
            if metric in ["OFFLINE_ACC", "OFFLINE_NDCG"]:
                future_incorrect_qs = {}
                for item_data in users_data:
                    user_id = item_data["user_id"]
                    valid_end_idx = item_data["valid_end_idx"]
                    seq_len = item_data["seq_len"]
                    question_seq = item_data["question_seq"][valid_end_idx:seq_len]
                    correctness_seq = item_data["correctness_seq"][valid_end_idx:seq_len]
                    future_incorrect_qs[user_id] = get_future_incorrect_questions(question_seq, correctness_seq)
                
                rec_ques_ = []
                incorrect_qs_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    incorrect_qs_.append(future_incorrect_qs[user_id])

                if metric == "OFFLINE_ACC":
                    performance[top_n][metric] = offline_acc(incorrect_qs_, rec_ques_)
                if metric == "OFFLINE_NDCG":
                    performance[top_n][metric] = offline_ndcg(incorrect_qs_, rec_ques_)
    return performance
