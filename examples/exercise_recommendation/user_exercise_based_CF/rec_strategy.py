def rec_method_0(users_history, similar_questions, top_n):
    rec_ques = {x["user_id"]: [] for x in users_history}
    for item_data in users_history:
        seq_len = item_data["seq_len"]
        question_seq = item_data["question_seq"][:seq_len]
        correct_seq = item_data["correct_seq"][:seq_len]
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
