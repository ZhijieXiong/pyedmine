import os
import random
from time import sleep
from tqdm import tqdm

from dspy_model import *

from edmine.utils.data_io import read_json


def predict_basic(dspy_lm, kt_data, question_meta, concept_meta, q2c, output_path, params):
    if not os.path.exists(output_path):
        prediction = {}
    else:
        prediction = read_json(output_path)

    num_evaluated = 0
    num2evaluate = params["num2evaluate"]
    max_history = params["max_history"]
    max_interval_time = max_history * 24 * 60 * 60
    seq_start = params["seq_start"]
    progress_bar = tqdm(total=num2evaluate)
    try:
        for item_data in kt_data:
            user_id = item_data["user_id"]
            if num_evaluated >= num2evaluate:
                break
            seq_len = item_data["seq_len"]
            i_tokens = 0
            o_tokens = 1
            for i in range(seq_start-1, seq_len):
                if f"{user_id}-{i}" in prediction:
                    continue

                q_id = item_data["question_seq"][i]
                q_text = question_meta[q_id]["text"]
                c_ids = q2c[q_id]
                correctness = item_data["correctness_seq"][i]
                current_time = item_data["time_seq"][i]
                # interaction_ids_用于报错情况
                interaction_ids_ = []
                used_history = []
                for j in list(range(i))[::-1]:
                    history_time = item_data["time_seq"][j]
                    interval_time = (current_time - history_time)
                    if interval_time >= max_interval_time:
                        continue
                    history_q_id = item_data["question_seq"][j]
                    history_c_ids = q2c[history_q_id]
                    s = f"{j}: {interval_time // (60 * 60 * 24)}; "
                    first_add = True
                    for history_c_id in history_c_ids:
                        s += concept_meta[history_c_id]["text"] + ", "
                        if (history_c_id in c_ids) and first_add:
                            interaction_ids_.append(j)
                            first_add = False
                    used_history.append(s[:-2])

                selected_history_interactions = dspy.Predict(SelectUserHistoryExercise)(
                    question=q_text,
                    history_interactions="\n".join(used_history)
                ).selected_history_interactions
                i_tokens += dspy_lm.history[-1]["input_tokens"]
                o_tokens += dspy_lm.history[-1]["output_tokens"]
                try:
                    # 解析selected_history_id
                    interaction_ids = list(map(lambda x: int(x.strip()), selected_history_interactions.split(",")))
                except:
                    # 如果解析报错，就直接用相同知识点的习题
                    interaction_ids = interaction_ids_
                interaction_ids = interaction_ids[:15]

                if len(interaction_ids) == 0:
                    continue
                sleep(1)

                selected_history = []
                for j in interaction_ids:
                    history_time = item_data["time_seq"][j]
                    interval_time = (current_time - history_time)
                    if interval_time >= max_interval_time:
                        continue
                    history_q_id = item_data["question_seq"][j]
                    history_correctness = item_data["correctness_seq"][j]
                    history_q_text = question_meta[history_q_id]["text"]
                    history_c_ids = q2c[history_q_id]
                    s = f"{interval_time // (60 * 60 * 24)}: {history_correctness}; {history_q_text}; "
                    for history_c_id in history_c_ids:
                        s += concept_meta[history_c_id]["text"] + ", "
                    selected_history.append(s[:-2])
                try:
                    predict_result = dspy.Predict(PredictUserAnswerCorrectness)(
                        question=q_text,
                        history_exercised_questions="\n".join(selected_history)
                    )
                    i_tokens += dspy_lm.history[-1]["input_tokens"]
                    o_tokens += dspy_lm.history[-1]["output_tokens"]
                    if "y" in predict_result.predict_result.lower():
                        predict_label = 1
                        predict_explanation = predict_result.predict_explanation
                    elif "n" in predict_result.predict_result.lower():
                        predict_label = 0
                        predict_explanation = predict_result.predict_explanation
                    else:
                        predict_label = random.choice([0, 1])
                        predict_explanation = ""
                except:
                    predict_label = random.choice([0, 1])
                    predict_explanation = ""

                prediction[f"{user_id}-{i}"] = {
                    "gt": correctness,
                    "pl": predict_label,
                    "pe": predict_explanation,
                    "i_tokens": i_tokens,
                    "o_tokens": o_tokens
                }
                num_evaluated += 1
                progress_bar.update(1)
                if num_evaluated >= num2evaluate:
                    break
        progress_bar.close()
    except:
        return prediction
    finally:
        return prediction


def predict_rag(dspy_lm, kt_data, question_meta, concept_meta, q2c, similar_questions, output_path, params):
    if not os.path.exists(output_path):
        prediction = {}
    else:
        prediction = read_json(output_path)

    num_evaluated = 0
    num2evaluate = params["num2evaluate"]
    max_history = params["max_history"]
    max_interval_time = max_history * 24 * 60 * 60
    seq_start = params["seq_start"]
    progress_bar = tqdm(total=num2evaluate)
    try:
        for item_data in kt_data:
            user_id = item_data["user_id"]
            if num_evaluated >= num2evaluate:
                break
            seq_len = item_data["seq_len"]
            i_tokens = 0
            o_tokens = 1
            for i in range(seq_start-1, seq_len):
                if f"{user_id}-{i}" in prediction:
                    continue

                q_id = item_data["question_seq"][i]
                similar_q_ids = similar_questions[q_id]
                q_text = question_meta[q_id]["text"]
                c_ids = q2c[q_id]
                correctness = item_data["correctness_seq"][i]
                current_time = item_data["time_seq"][i]
                # interaction_ids_用于报错情况
                interaction_ids_ = []
                similar_interaction_ids = []
                used_history = []
                for j in list(range(i))[::-1]:
                    history_time = item_data["time_seq"][j]
                    interval_time = (current_time - history_time)
                    if interval_time >= max_interval_time:
                        continue
                    history_q_id = item_data["question_seq"][j]
                    if history_q_id in similar_q_ids:
                        similar_interaction_ids.append(j)
                    history_c_ids = q2c[history_q_id]
                    s = f"{j}: {interval_time // (60 * 60 * 24)}; "
                    first_add = True
                    for history_c_id in history_c_ids:
                        s += concept_meta[history_c_id]["text"] + ", "
                        if (history_c_id in c_ids) and first_add:
                            interaction_ids_.append(j)
                            first_add = False
                    used_history.append(s[:-2])

                selected_history_interactions = dspy.Predict(SelectUserHistoryExercise)(
                    question=q_text,
                    history_interactions="\n".join(used_history)
                ).selected_history_interactions
                i_tokens += dspy_lm.history[-1]["input_tokens"]
                o_tokens += dspy_lm.history[-1]["output_tokens"]
                try:
                    # 解析selected_history_id
                    interaction_ids = list(map(lambda x: int(x.strip()), selected_history_interactions.split(",")))
                except:
                    # 如果解析报错，就直接用相同知识点的习题
                    interaction_ids = interaction_ids_
                if len(interaction_ids) < 10:
                    interaction_ids = list(set(interaction_ids).union(set(similar_interaction_ids[:5])))
                    interaction_ids = sorted(interaction_ids, reverse=True)

                if len(interaction_ids) == 0:
                    continue
                sleep(1)

                selected_history = []
                for j in interaction_ids:
                    history_time = item_data["time_seq"][j]
                    interval_time = (current_time - history_time)
                    if interval_time >= max_interval_time:
                        continue
                    history_q_id = item_data["question_seq"][j]
                    history_correctness = item_data["correctness_seq"][j]
                    history_q_text = question_meta[history_q_id]["text"]
                    history_c_ids = q2c[history_q_id]
                    s = f"{interval_time // (60 * 60 * 24)}: {history_correctness}; {history_q_text}; "
                    for history_c_id in history_c_ids:
                        s += concept_meta[history_c_id]["text"] + ", "
                    selected_history.append(s[:-2])
                try:
                    predict_result = dspy.Predict(PredictUserAnswerCorrectness)(
                        question=q_text,
                        history_exercised_questions="\n".join(selected_history)
                    )
                    i_tokens += dspy_lm.history[-1]["input_tokens"]
                    o_tokens += dspy_lm.history[-1]["output_tokens"]
                    if "y" in predict_result.predict_result.lower():
                        predict_label = 1
                        predict_explanation = predict_result.predict_explanation
                    elif "n" in predict_result.predict_result.lower():
                        predict_label = 0
                        predict_explanation = predict_result.predict_explanation
                    else:
                        predict_label = random.choice([0, 1])
                        predict_explanation = ""
                except:
                    predict_label = random.choice([0, 1])
                    predict_explanation = ""

                prediction[f"{user_id}-{i}"] = {
                    "gt": correctness,
                    "pl": predict_label,
                    "pe": predict_explanation,
                    "i_tokens": i_tokens,
                    "o_tokens": o_tokens
                }
                num_evaluated += 1
                progress_bar.update(1)
                if num_evaluated >= num2evaluate:
                    break
        progress_bar.close()
    except:
        return prediction
    finally:
        return prediction
