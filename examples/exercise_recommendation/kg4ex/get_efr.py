import argparse
import os

import config
from utils import *

from edmine.utils.parse import q2c_from_q_table
from edmine.data.FileManager import FileManager
from edmine.utils.data_io import read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 遗忘模型参数
    parser.add_argument("--theta", type=float, default=0.2)
    # 数据配置
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--user_data_file_name", type=str, default="assist2009_user_data.txt")
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    kg4ex_dir = os.path.join(setting_dir, "KG4EX")
    if not os.path.exists(kg4ex_dir):
        os.mkdir(kg4ex_dir)
    dataset_name = params["dataset_name"]
    theta = params["theta"]
    efr_train_path = os.path.join(kg4ex_dir, f"{dataset_name}_efr_{theta}_train.txt")
    efr_valid_path = os.path.join(kg4ex_dir, f"{dataset_name}_efr_{theta}_valid.txt")
    efr_test_path = os.path.join(kg4ex_dir, f"{dataset_name}_efr_{theta}_test.txt")
    
    if (not os.path.exists(efr_train_path)) or (not os.path.exists(efr_valid_path)) or (not os.path.exists(efr_test_path)):  
        users_data = read_kt_file(os.path.join(setting_dir, params["user_data_file_name"]))
        Q_table = file_manager.get_q_table(params["dataset_name"])
        question2concept = q2c_from_q_table(Q_table)

        # 测试集
        efr_train = []
        for user_data in users_data:
            efr_train.append(
                get_efr(user_data["user_id"], get_last_frkc(user_data, question2concept, Q_table.shape[1], params["theta"]),
                        question2concept)
            )

        # 验证集
        for user_data in users_data:
            valid_end_idx = user_data["valid_end_idx"]
            user_data["seq_len"] = valid_end_idx
            for k, v in user_data.items():
                if type(v) is list:
                    for i, _ in enumerate(v[valid_end_idx:]):
                        v[valid_end_idx + i] = 0
        efr_valid = []
        for user_data in users_data:
            efr_valid.append(
                get_efr(user_data["user_id"], get_last_frkc(user_data, question2concept, Q_table.shape[1], params["theta"]),
                        question2concept)
            )

        # 训练集
        for user_data in users_data:
            train_end_idx = user_data["train_end_idx"]
            valid_end_idx = user_data["valid_end_idx"]
            user_data["seq_len"] = train_end_idx
            for k, v in user_data.items():
                if type(v) is list:
                    for i, _ in enumerate(v[train_end_idx:valid_end_idx]):
                        v[train_end_idx + i] = 0
        efr_test = []
        for user_data in users_data:
            efr_test.append(
                get_efr(user_data["user_id"], get_last_frkc(user_data, question2concept, Q_table.shape[1], params["theta"]),
                        question2concept)
            )

        save_data(efr_train_path, efr_train)
        save_data(efr_valid_path, efr_valid)
        save_data(efr_test_path, efr_test)
