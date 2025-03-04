def rec_method_based_on_que_sim(users_history, similar_questions, top_n):
    rec_ques = {x["user_id"]: [] for x in users_history}
    for item_data in users_history:
        seq_len = item_data["seq_len"]
        question_seq = item_data["question_seq"][:seq_len]
        correct_seq = item_data["correctness_seq"][:seq_len]
        answered_ques = set(question_seq)
        target_question = question_seq[-1]
        if sum(correct_seq) != seq_len:
            for q_id, correctness in zip(question_seq[::-1], correct_seq[::-1]):
                if correctness == 0:
                    target_question = q_id
                    break

        similar_ques_sorted = similar_questions[target_question]
        num_rec = 0
        for q_id in similar_ques_sorted:
            if num_rec >= top_n:
                break
            if q_id in answered_ques:
                continue
            num_rec += 1
            rec_ques[item_data["user_id"]].append(q_id)

    return rec_ques


def rec_method_based_on_user_sim(users_history, similar_users, question_diff, th, top_n):
    users_answered_ques = {}
    for item_data in users_history:
        users_answered_ques[item_data["user_id"]] = set(item_data["question_seq"][:item_data["seq_len"]])

    rec_ques = {x["user_id"]: [] for x in users_history}
    for item_data in users_history:
        user_id = item_data["user_id"]
        seq_len = item_data["seq_len"]
        question_seq = item_data["question_seq"][:seq_len]
        correct_seq = item_data["correctness_seq"][:seq_len]
        answered_ques = set(question_seq)
        average_diff = 1 - sum(correct_seq) / seq_len
        while len(rec_ques[user_id]) < top_n:
            # 如果阈值过小，可能不能满足推荐top n个习题，加大阈值
            th += 0.05
            for sim_user_id in similar_users[user_id]:
                if sim_user_id not in users_answered_ques:
                    continue
                for q_id in (users_answered_ques[sim_user_id] - answered_ques):
                    q_diff = question_diff[q_id]
                    if abs(average_diff - q_diff) < 0.1:
                        rec_ques[user_id].append(q_id)
                    if len(rec_ques[user_id]) >= top_n:
                        break
                if len(rec_ques[user_id]) >= top_n:
                    break

    return rec_ques