import argparse
import os

import config

from edmine.data.FileManager import FileManager
from edmine.utils.data_io import write_kt_file, read_kt_file
from edmine.utils.parse import q2c_from_q_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--test_file_name", type=str, default="xes3g5m_test.txt")
    parser.add_argument("--num_data", type=int, default=100)
    args = parser.parse_args()
    params = vars(args)
    
    file_namager = FileManager(config.FILE_MANAGER_ROOT)
    setting_dir = file_namager.get_setting_dir(params["setting_name"])
    q_table = file_namager.get_q_table(params["dataset_name"])
    q2c = q2c_from_q_table(q_table)
    test_data = read_kt_file(os.path.join(setting_dir, params["test_file_name"]))
    
    candiate = []
    for user_data in test_data:
        seq_len = user_data["seq_len"]
        if seq_len < 200:
            continue
        num_correct = sum(user_data["correctness_seq"])
        if (num_correct < 50) or (num_correct > 150):
            continue
        concept_exercised = set()
        for q_id in user_data["question_seq"]:
            c_ids = q2c[q_id]
            concept_exercised.update(c_ids)
        user_data["num_c_exercised"] = len(concept_exercised)
        candiate.append(user_data)
        
    candiate_sorted = sorted(candiate, key=lambda x: x["num_c_exercised"], reverse=True)
    final_data = candiate_sorted[:params["num_data"]]
    save_path = os.path.join(setting_dir, f"{params['dataset_name']}-subtest-{params['num_data']}.txt")
    write_kt_file(final_data, save_path)