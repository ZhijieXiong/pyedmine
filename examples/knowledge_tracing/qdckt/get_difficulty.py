import argparse
import os
from collections import defaultdict

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file, write_json
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


def parse_difficulty(kt_data_, num_question_diff, num_question):
    questions_frequency, questions_accuracy = defaultdict(int), defaultdict(int)
    n_sum = 0
    n_correct = 0
    for item_data in kt_data_:
        seq_len = item_data["seq_len"]
        question_seq = item_data["question_seq"]
        correctness_seq = item_data["correctness_seq"]
        n_sum += seq_len
        n_correct += sum(correctness_seq)
        for i in range(seq_len):
            q_id = question_seq[i]
            questions_frequency[q_id] += 1
            questions_accuracy[q_id] += correctness_seq[i]

    ave_acc = n_correct / n_sum
    for q_id in range(num_question):
        if questions_frequency[q_id] == 0:
            diff = 1 - ave_acc
        else:
            diff = 1 - (questions_accuracy[q_id] + 5 * ave_acc) / (questions_frequency[q_id] + 5)
        questions_accuracy[q_id] = int((num_question_diff - 1) * diff)
    return questions_accuracy
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    parser.add_argument("--num_question_diff", type=int, default=100)
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    dimkt_dir = os.path.join(setting_dir, "QDCKT")
    if not os.path.exists(dimkt_dir):
        os.mkdir(dimkt_dir)
    train_file_name = params["train_file_name"]
    save_path = os.path.join(dimkt_dir, train_file_name.replace(".txt", "_qdckt_diff.json"))
    if not os.path.exists(save_path):
        kt_data = read_kt_file(os.path.join(setting_dir, train_file_name))
        q_table = file_manager.get_q_table(params["dataset_name"])
        q2c = q2c_from_q_table(q_table)
        num_q = q_table.shape[0]
        
        difficulty_info = {
            "num_question_diff": params["num_question_diff"],
            "question_difficulty": parse_difficulty(kt_data, params["num_question_diff"], num_q)
        }
        write_json(difficulty_info, save_path)
