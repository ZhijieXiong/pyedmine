import argparse
import os
import numpy as np

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file, write_json
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


def get_statics4lbkt(kt_daata_, use_use_time_first=True):
    use_time_dict = {}
    num_attempt_dict = {}
    num_hint_dict = {}
    for item_data in kt_daata_:
        for i in range(item_data["seq_len"]):
            q_id = item_data["question_seq"][i]
            if not use_use_time_first:
                # 有些数据集没有use_time_first_attempt，所以使用use_time代替
                use_time_first = item_data["use_time_seq"][i]
            else:
                use_time_first = item_data["use_time_first_seq"][i]
            num_attempt = item_data["num_attempt_seq"][i]
            num_hint = item_data["num_hint_seq"][i]

            if use_time_first > 0:
                use_time_dict.setdefault(q_id, [])
                use_time_dict[q_id].append(use_time_first)

            if num_attempt >= 0:
                num_attempt_dict.setdefault(q_id, [])
                num_attempt_dict[q_id].append(num_attempt)

            if num_hint >= 0:
                num_hint_dict.setdefault(q_id, [])
                num_hint_dict[q_id].append(num_hint)
    use_time_mean_dict = {k: np.mean(v) for k, v in use_time_dict.items()}
    use_time_std_dict = {k: np.var(v) for k, v in use_time_dict.items()}
    num_attempt_mean_dict = {k: np.mean(v) for k, v in num_attempt_dict.items()}
    num_hint_mean_dict = {k: np.mean(v) for k, v in num_hint_dict.items()}

    return use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    lbkt_dir = os.path.join(setting_dir, "LBKT")
    if not os.path.exists(lbkt_dir):
        os.mkdir(lbkt_dir)
    train_file_name = params["train_file_name"]
    save_path = os.path.join(lbkt_dir, train_file_name.replace(".txt", "_lbkt_statics.json"))
    if not os.path.exists(save_path):
        kt_data = read_kt_file(os.path.join(setting_dir, train_file_name))
        q_table = file_manager.get_q_table(params["dataset_name"])
        q2c = q2c_from_q_table(q_table)
        num_q, num_c = q_table.shape[0], q_table.shape[1]
        
        factors = get_statics4lbkt(kt_data, params["dataset_name"] in ["assist2009", "assist2012", "junyi2015"])
        write_json({
            "use_time_mean_dict": factors[0],
            "use_time_std_dict": factors[1],
            "num_attempt_mean_dict": factors[2],
            "num_hint_mean_dict": factors[3]
        }, save_path)