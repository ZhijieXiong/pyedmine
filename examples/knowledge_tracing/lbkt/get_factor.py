import argparse
import os
import numpy as np
from scipy.stats import norm
from scipy.stats import poisson

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file, read_json, write_kt_file
from edmine.data.FileManager import FileManager


def generate_factor(kt_data_, use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict, use_use_time_first=True):
    max_seq_len = len(kt_data_[0]["mask_seq"])
    # 需要考虑统计信息是从训练集提取的，在测试集和验证集中有些习题没在训练集出现过，对于这些习题，就用训练集的平均代替
    use_time_mean4unseen = sum(use_time_mean_dict.values()) / len(use_time_std_dict)
    use_time_std4unseen = sum(use_time_std_dict.values()) / len(use_time_std_dict)
    num_attempt_mean4unseen = sum(num_attempt_mean_dict.values()) / len(num_attempt_mean_dict)
    num_hint_mean4unseen = sum(num_hint_mean_dict.values()) / len(num_hint_mean_dict)
    for item_data in kt_data_:
        time_factor_seq = []
        attempt_factor_seq = []
        hint_factor_seq = []
        seq_len = item_data["seq_len"]
        for i in range(seq_len):
            q_id = item_data["question_seq"][i]
            use_time_mean = use_time_mean_dict.get(q_id, use_time_mean4unseen)
            use_time_std = use_time_std_dict.get(q_id, use_time_std4unseen)
            num_attempt_mean = num_attempt_mean_dict.get(q_id, num_attempt_mean4unseen)
            num_hint_mean = num_hint_mean_dict.get(q_id, num_hint_mean4unseen)

            if not use_use_time_first:
                use_time_first = item_data["use_time_seq"][i]
            else:
                use_time_first = item_data["use_time_first_seq"][i]
            # 有些数据集use time first <= 0 （前端均处理为0）
            if use_time_first == 0:
                use_time_first = int(use_time_mean4unseen)
            time_factor = 1 if (use_time_std == 0) else norm(use_time_mean, use_time_std).cdf(np.log(use_time_first))
            time_factor_seq.append(time_factor)

            num_attempt = item_data["num_attempt_seq"][i]
            if num_attempt < 0:
                num_attempt = int(num_attempt_mean4unseen)
            attempt_factor = 1 - poisson(num_attempt_mean).cdf(num_attempt - 1)
            attempt_factor_seq.append(attempt_factor)

            num_hint = item_data["num_hint_seq"][i]
            if num_attempt < 0:
                num_hint = int(num_hint_mean4unseen)
            hint_factor = 1 - poisson(num_hint_mean).cdf(num_hint - 1)
            hint_factor_seq.append(hint_factor)

            if (use_time_first <= 0) or (str(time_factor) == "nan"):
                print(f"time error: {use_time_first}, {time_factor}")
            if str(attempt_factor) == "nan":
                print(f"time error: {num_attempt}, {attempt_factor}")
            if str(hint_factor) == "nan":
                print(f"time error: {num_hint}, {hint_factor}")
        item_data["time_factor_seq"] = time_factor_seq + [0.] * (max_seq_len - seq_len)
        item_data["attempt_factor_seq"] = attempt_factor_seq + [0.] * (max_seq_len - seq_len)
        item_data["hint_factor_seq"] = hint_factor_seq + [0.] * (max_seq_len - seq_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--statics_file_name", type=str, default="assist2009_train_fold_0_lbkt_statics.json")
    parser.add_argument("--target_file_name", type=str, default="assist2009_train_fold_0.txt")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    lbkt_dir = os.path.join(setting_dir, "LBKT")
    target_file_name = params["target_file_name"]
    save_path = os.path.join(lbkt_dir, target_file_name)
    if not os.path.exists(save_path):
        statics_path = os.path.join(lbkt_dir, params["statics_file_name"])
        statics = read_json(statics_path)
        use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict = {}, {}, {}, {}
        for k, v in statics["use_time_mean_dict"].items():
            use_time_mean_dict[int(k)] = v
        for k, v in statics["use_time_std_dict"].items():
            use_time_std_dict[int(k)] = v
        for k, v in statics["num_attempt_mean_dict"].items():
            num_attempt_mean_dict[int(k)] = v
        for k, v in statics["num_hint_mean_dict"].items():
            num_hint_mean_dict[int(k)] = v
        kt_data = read_kt_file(os.path.join(setting_dir, target_file_name))
        generate_factor(kt_data, use_time_mean_dict, use_time_std_dict, num_attempt_mean_dict, num_hint_mean_dict,
                        params["dataset_name"] in ["assist2009", "assist2012", "junyi2015"])
        write_kt_file(kt_data, save_path)
        