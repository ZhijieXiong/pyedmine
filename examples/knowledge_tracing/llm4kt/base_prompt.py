import argparse
import os
import dspy
import inspect
import random
from tqdm import tqdm

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file, read_json, read_csv, write_json
from edmine.data.FileManager import FileManager
from edmine.llm.dspy.remote_llm.GLM import GLM
from edmine.llm.dspy.remote_llm.BaiLian import BaiLian
from edmine.utils.parse import q2c_from_q_table


class SelectUserHistoryExercise(dspy.Signature):
    """你需要根据学生的历史练习记录，推断ta是否能做对指定习题，首先请根据待预测的习题，从学生的历史练习中选择你认为对预测有帮助的记录，其中学生的历史练习记录将会以`interaction_id1: practice time; related concepts\ninteraction_id2: practice time; related concepts\n...`的格式展示，你只需要按照`qid1,qid2, ...`的格式回复哪些习题练习记录是你想要查看的即可"""
    question = dspy.InputField(desc="待预测的习题")
    history_interactions = dspy.InputField(desc="学生历史练习记录的信息，每一行的格式为`interaction_id1: practice time; related concepts`，其中practice time表示这个练习记录是发生在多久之前，单位为天，related concepts表示这次练习习题关联的知识点")
    selected_history_interactions = dspy.OutputField(desc="你认为对于预测学生是否能做对习题有帮助的习题练习记录，格式为`interaction_id1,interaction_id1, ...`，例如`1,3`表示查看interaction_id为1和2的学生历史练习记录")


class PredictUserAnswerCorrectness(dspy.Signature):
    """你需要根据学生的历史练习记录，推断ta是否能做对指定习题"""
    question = dspy.InputField(desc="待预测的习题")
    history_exercised_questions = dspy.InputField(desc="学生历史练习记录的信息，每一行的格式为`practice time: answer result; question text; related concepts`，其中answer result为1表示学生做对这道习题，为0表示做错，practice time表示这个练习记录是发生在多久之前，单位为天，question text是习题的文本信息，related concepts表示这次练习习题关联的知识点，例如`3: 0; question text; related concepts`表示学生3天前做错了这道习题")
    predict_result = dspy.OutputField(desc="基于学生的历史练习记录，你认为学生是否能做对习题，只需要回答`Y`或者`N`，Y表示认为能做对，N表示会做错")
    predict_explanation = dspy.OutputField(desc="你对于预测结果的解释")


def base_prompt_evaluate(kt_data_, question_meta_, concept_meta_, q2c_, output_path_, params_):
    if not os.path.exists(output_path_):
        prediction_ = {}
    else:
        prediction_ = read_json(output_path_)

    num_evaluated = 0
    num2evaluate = params_["num2evaluate"]
    max_history = params_["max_history"]
    max_interval_time = max_history * 24 * 60 * 60
    seq_start = params_["seq_start"]
    progress_bar = tqdm(total=num2evaluate)
    try:
        for item_data in kt_data_:
            user_id = item_data["user_id"]
            if num_evaluated >= num2evaluate:
                break
            seq_len = item_data["seq_len"]
            for i in range(seq_start-1, seq_len):
                if f"{user_id}-{i}" in prediction_:
                    continue

                q_id = item_data["question_seq"][i]
                q_text = question_meta_[q_id]["text"]
                c_ids = q2c_[q_id]
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
                    history_c_ids = q2c_[history_q_id]
                    s = f"{j}: {interval_time // (60 * 60 * 24)}; "
                    first_add = True
                    for history_c_id in history_c_ids:
                        s += concept_meta_[history_c_id]["text"] + ", "
                        if (history_c_id in c_ids) and first_add:
                            interaction_ids_.append(j)
                            first_add = False
                    used_history.append(s[:-2])

                selected_history_interactions = dspy.Predict(SelectUserHistoryExercise)(
                    question=q_text,
                    history_interactions="\n".join(used_history)
                ).selected_history_interactions
                try:
                    # 解析selected_history_id
                    interaction_ids = list(map(lambda x: int(x.strip()), selected_history_interactions.split(",")))
                except:
                    # 如果解析报错，就直接用相同知识点的习题
                    interaction_ids = interaction_ids_

                if len(interaction_ids) == 0:
                    continue

                selected_history = []
                for j in interaction_ids:
                    history_time = item_data["time_seq"][j]
                    interval_time = (current_time - history_time)
                    if interval_time >= max_interval_time:
                        continue
                    history_q_id = item_data["question_seq"][j]
                    history_correctness = item_data["correctness_seq"][j]
                    history_q_text = question_meta_[history_q_id]["text"]
                    history_c_ids = q2c_[history_q_id]
                    s = f"{interval_time // (60 * 60 * 24)}: {history_correctness}; {history_q_text}; "
                    for history_c_id in history_c_ids:
                        s += concept_meta_[history_c_id]["text"] + ", "
                    selected_history.append(s[:-2])
                try:
                    predict_result = dspy.Predict(PredictUserAnswerCorrectness)(
                        question=q_text,
                        history_exercised_questions="\n".join(selected_history)
                    )
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

                prediction_[f"{user_id}-{i}"] = {
                    "gt": correctness,
                    "pl": predict_label,
                    "pe": predict_explanation
                }
                num_evaluated += 1
                progress_bar.update(1)
                if num_evaluated >= num2evaluate:
                    break
        progress_bar.close()
    except:
        return prediction_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="qwen-long")
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--test_file_name", type=str, default="xes3g5m-subtest-100.txt")
    parser.add_argument("--seq_start", type=int, default=171)
    parser.add_argument("--max_history", type=int, default=90, help="unit: day")
    parser.add_argument("--num2evaluate", type=int, default=100)
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(FILE_MANAGER_ROOT)
    processed_dir = file_manager.get_preprocessed_dir(params["dataset_name"])
    setting_dir = file_manager.get_setting_dir(params["setting_name"])

    q_table = file_manager.get_q_table(params["dataset_name"])
    q2c = q2c_from_q_table(q_table)
    kt_data = read_kt_file(os.path.join(setting_dir, params["test_file_name"]))
    question_id_map = read_csv(os.path.join(processed_dir, "question_id_map.csv"))
    concept_id_map = read_csv(os.path.join(processed_dir, "concept_id_map.csv"))
    question_id_map = question_id_map.to_dict()
    concept_id_map = concept_id_map.to_dict()
    question_meta = {
        q_id: {
            "text": q_text
        } for q_id, q_text in zip(question_id_map["mapped_id"].values(), question_id_map["text"].values())
    }
    concept_meta = {
        c_id: {
            "text": c_text
        } for c_id, c_text in zip(concept_id_map["mapped_id"].values(), concept_id_map["text"].values())
    }

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    # 选择LLM
    if args.llm in ["glm-4-plus"]:
        dspy_lm = GLM(f"zhipu/{args.llm}")
    elif args.llm in ["qwen-plus", "qwen-long"]:
        dspy_lm = BaiLian(f"bailian/{args.llm}")
    else:
        raise NotImplementedError()
    dspy.configure(lm=dspy_lm)

    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(current_dir,
                               f"output/{args.llm}_{args.test_file_name.replace('.txt', '')}_{args.seq_start}_{args.max_history}.json")
    prediction = base_prompt_evaluate(kt_data, question_meta, concept_meta, q2c, output_path, params)
    write_json(prediction, output_path)

