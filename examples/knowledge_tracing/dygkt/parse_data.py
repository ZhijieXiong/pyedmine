import argparse
import os
from collections import defaultdict

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file, write_json
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


def parse_difficulty(kt_data_, question2concept, func_params):
    num_min_question = 5
    num_min_concept = 5
    num_question_diff = 100
    num_concept_diff = 100
    num_concept = func_params["num_concept"]
    num_question = func_params["num_question"]

    questions_frequency, concepts_frequency = defaultdict(int), defaultdict(int)
    questions_accuracy, concepts_accuracy = defaultdict(int), defaultdict(int)
    # 用于给统计信息不足的知识点和习题赋值难度
    num_few_shot_q = 0
    num_few_shot_c = 0
    for item_data in kt_data_:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            questions_frequency[q_id] += 1
            c_ids = question2concept[q_id]
            for c_id in c_ids:
                concepts_frequency[c_id] += 1
            questions_accuracy[q_id] += item_data["correctness_seq"][i]
            for c_id in c_ids:
                concepts_accuracy[c_id] += item_data["correctness_seq"][i]

    for q_id in range(num_question):
        if questions_frequency[q_id] < num_min_question:
            questions_accuracy[q_id] = num_question_diff + num_few_shot_q
            num_few_shot_q += 1
        else:
            questions_accuracy[q_id] = int(
                (num_question_diff - 1) * questions_accuracy[q_id] / questions_frequency[q_id])
    for c_id in range(num_concept):
        if concepts_frequency[c_id] < num_min_concept:
            concepts_accuracy[c_id] = num_concept_diff + num_few_shot_c
            num_few_shot_c += 1
        else:
            concepts_accuracy[c_id] = int((num_concept_diff - 1) * concepts_accuracy[c_id] / concepts_frequency[c_id])

    return questions_accuracy, concepts_accuracy
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")

    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    dimkt_dir = os.path.join(setting_dir, "DyGKT")
    if not os.path.exists(dimkt_dir):
        os.mkdir(dimkt_dir)
    train_file_name = params["train_file_name"]
    save_path = os.path.join(dimkt_dir, train_file_name.replace(".txt", "_dyckt.json"))
    if not os.path.exists(save_path):
        kt_data = read_kt_file(os.path.join(setting_dir, train_file_name))
        q_table = file_manager.get_q_table(params["dataset_name"])
        q2c = q2c_from_q_table(q_table)
        num_q, num_c = q_table.shape[0], q_table.shape[1]
        
        q_acc, c_acc = parse_difficulty(kt_data, q2c, {
            "num_min_question": params["num_min_question"],
            "num_min_concept": params["num_min_concept"],
            "num_question_diff": params["num_question_diff"],
            "num_concept_diff": params["num_concept_diff"],
            "num_concept": num_c,
            "num_question": num_q
        })
        difficulty_info = {
            "question_difficulty": q_acc, 
            "concept_difficulty": c_acc
        }
        write_json(difficulty_info, save_path)