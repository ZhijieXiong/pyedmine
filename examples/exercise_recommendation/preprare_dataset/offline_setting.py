import argparse
import os
import random

import config

from edmine.data.FileManager import FileManager
from edmine.utils.data_io import read_kt_file, write_kt_file, write_cd_file


if __name__ == "__main__":
    # 选择知识追踪实验的测试集作为习题推荐的数据集，其中每个用户的前70%数据作为训练集，后30%数据作为测试集
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)

    er_setting = {
        "name": "ER_offline_setting",
        "drop_seq_len": 10,
        "test_radio": 0.3,
    }
    cd_setting = {
        "name": "CD_setting4ER_offline_setting",
        "drop_seq_len": 10,
        "valid_radio": 0.2,
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(er_setting["name"], er_setting)
    kt_setting_dir = file_manager.get_setting_dir("pykt_setting")
    kt_data_train = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_train.txt"))
    kt_data_valid = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_valid.txt"))
    kt_data_test = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_test.txt"))
    
    # 不论是er data还是cd data都不要对user id进行重映射，保持原来的id，方便后续习题推荐处理
    er_data = []
    num_user = len(kt_data_train + kt_data_valid + kt_data_test)
    for user_data in kt_data_test:
        seq_len = user_data["seq_len"]
        if seq_len < 10:
            continue
        user_data["train_end_idx"] = int(seq_len * 0.5)
        user_data["valid_end_idx"] = int(seq_len * 0.7)
        er_data.append(user_data)
    
    # 用于训练认知诊断模型，为习题推荐模型服务
    kt_data_train_valid = []
    for user_data in (kt_data_train + kt_data_valid):
        seq_len = user_data["seq_len"]
        if seq_len < 10:
            continue
        kt_data_train_valid.append(user_data)

    cd_data_train = []
    cd_data_valid = []
    for user_data in er_data:
        valid_end_idx = user_data["valid_end_idx"]
        i_list = list(range(valid_end_idx))
        random.shuffle(i_list)
        num_train = int(valid_end_idx * 0.8)
        for i in i_list[:num_train]:
            interaction_data = {
                "user_id": user_data["user_id"],
                "question_id": user_data["question_seq"][i],
                "correctness": user_data["correctness_seq"][i],
            }
            if "use_time_seq" in user_data:
                interaction_data["use_time"] = user_data["use_time_seq"][i]
            cd_data_train.append(interaction_data)
        for i in i_list[num_train:]:
            interaction_data = {
                "user_id": user_data["user_id"],
                "question_id": user_data["question_seq"][i],
                "correctness": user_data["correctness_seq"][i],
            }
            if "use_time_seq" in user_data:
                interaction_data["use_time"] = user_data["use_time_seq"][i]
            cd_data_valid.append(interaction_data)
    for user_data in kt_data_train_valid:
        seq_len = user_data["seq_len"]
        i_list = list(range(seq_len))
        random.shuffle(i_list)
        num_train = int(seq_len * 0.8)
        for i in i_list[:num_train]:
            interaction_data = {
                "user_id": user_data["user_id"],
                "question_id": user_data["question_seq"][i],
                "correctness": user_data["correctness_seq"][i],
            }
            if "use_time_seq" in user_data:
                interaction_data["use_time"] = user_data["use_time_seq"][i]
            cd_data_train.append(interaction_data)
        for i in i_list[num_train:]:
            interaction_data = {
                "user_id": user_data["user_id"],
                "question_id": user_data["question_seq"][i],
                "correctness": user_data["correctness_seq"][i],
            }
            if "use_time_seq" in user_data:
                interaction_data["use_time"] = user_data["use_time_seq"][i]
            cd_data_valid.append(interaction_data)

    file_manager.add_new_setting(cd_setting["name"], cd_setting)
    cd_setting_dir = file_manager.get_setting_dir(cd_setting["name"])
    write_cd_file(cd_data_train, os.path.join(cd_setting_dir, f"{params['dataset_name']}_train.txt"))
    write_cd_file(cd_data_valid, os.path.join(cd_setting_dir, f"{params['dataset_name']}_valid.txt"))
    with open(os.path.join(cd_setting_dir, f"{params['dataset_name']}_statics.txt"), "w") as f:
        f.write(f"num of user: {num_user}")

    er_setting_dir = file_manager.get_setting_dir(er_setting["name"])
    write_kt_file(er_data, os.path.join(er_setting_dir, f"{params['dataset_name']}_user_data.txt"))
