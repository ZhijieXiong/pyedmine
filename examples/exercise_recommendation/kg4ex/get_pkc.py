import argparse
import os

from config import config_roster
from utils import *

from edmine.roster.DLKTRoster import DLKTRoster
from edmine.utils.data_io import read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据配置
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--user_data_file_name", type=str, default="assist2009_user_data.txt")
    # 加载KT模型
    parser.add_argument("--model_dir_name", type=str, help="模型文件夹名",
                        default=r"DKT_KG4EX@@pykt_setting@@assist2009_train@@seed_0@@2025-03-12@11-32-04")
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # batch size
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_roster(params)
    kt_roster = DLKTRoster(global_params, global_objects)

    file_manager = global_objects["file_manager"]
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    kg4ex_dir = os.path.join(setting_dir, "kg4ex")
    if not os.path.exists(kg4ex_dir):
        os.mkdir(kg4ex_dir)
    dataset_name = params["dataset_name"]
    pkc_train_path = os.path.join(kg4ex_dir, f"{dataset_name}_pkc_train.txt")
    pkc_valid_path = os.path.join(kg4ex_dir, f"{dataset_name}_pkc_valid.txt")
    pkc_test_path = os.path.join(kg4ex_dir, f"{dataset_name}_pkc_test.txt")
    
    if (not os.path.exists(pkc_train_path)) or (not os.path.exists(pkc_valid_path)) or (not os.path.exists(pkc_test_path)):  
        users_data = read_kt_file(os.path.join(setting_dir, params["user_data_file_name"]))

        # 测试集
        batches_train = data2batches(users_data, params["batch_size"])
        mlkc_test = get_mlkc(kt_roster, batches_train)

        # 验证集
        for user_data in users_data:
            valid_end_idx = user_data["valid_end_idx"]
            user_data["seq_len"] = valid_end_idx
            for k, v in user_data.items():
                if type(v) is list:
                    for i, _ in enumerate(v[valid_end_idx:]):
                        v[valid_end_idx + i] = 0
        batches_valid = data2batches(users_data, params["batch_size"])
        mlkc_valid = get_mlkc(kt_roster, batches_valid)

        # 训练集
        for user_data in users_data:
            train_end_idx = user_data["train_end_idx"]
            valid_end_idx = user_data["valid_end_idx"]
            user_data["seq_len"] = train_end_idx
            for k, v in user_data.items():
                if type(v) is list:
                    for i, _ in enumerate(v[train_end_idx:valid_end_idx]):
                        v[train_end_idx + i] = 0
        batches_test = data2batches(users_data, params["batch_size"])
        mlkc_train = get_mlkc(kt_roster, batches_test)

        save_data(pkc_train_path, mlkc_train)
        save_data(pkc_valid_path, mlkc_valid)
        save_data(pkc_test_path, mlkc_test)
