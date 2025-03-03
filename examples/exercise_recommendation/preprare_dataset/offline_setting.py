import argparse
import os

import config

from edmine.data.FileManager import FileManager

from edmine.utils.data_io import read_kt_file, write_kt_file


if __name__ == "__main__":
    # 选择知识追踪实验的测试集作为习题推荐的数据集，其中每个用户的前70%数据作为训练集，后30%数据作为测试集
    parser = argparse.ArgumentParser()

    # device配置
    parser.add_argument("--kt_setting_name", type=str, default="pykt_setting")
    parser.add_argument("--kt_test_file_name", type=str, default="assist2009_test.txt")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)

    setting = {
        "name": "ER_offline_setting",
        "drop_seq_len": 10,
        "test_radio": 0.3,
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(setting["name"], setting)
    kt_setting_dir = file_manager.get_setting_dir(params["kt_setting_name"])
    data = read_kt_file(os.path.join(kt_setting_dir, params["kt_test_file_name"]))
    er_data = []
    for user_data in data:
        seq_len = user_data["seq_len"]
        if seq_len < 10:
            continue

        user_data["train_end_idx"] = int(seq_len * 0.5)
        user_data["valid_end_idx"] = int(seq_len * 0.7)
        er_data.append(user_data)

    setting_dir = file_manager.get_setting_dir(setting["name"])
    write_kt_file(er_data, os.path.join(setting_dir, f"{params['dataset_name']}_user_data.txt"))
