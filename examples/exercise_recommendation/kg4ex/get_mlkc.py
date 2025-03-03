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
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--user_data_file_name", type=str, default="statics2011_user_data.txt")
    # 加载KT模型
    parser.add_argument("--model_dir_name", type=str, help="绝对路径",
                        default=r"qDKT@@pykt_setting@@statics2011_train_fold_0@@seed_0@@2025-03-02@14-48-19")
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

    dataset_name = params["dataset_name"]
    save_data(os.path.join(setting_dir, f"{dataset_name}_qdkt_mlkc_train.txt"), mlkc_train)
    save_data(os.path.join(setting_dir, f"{dataset_name}_qdkt_mlkc_valid.txt"), mlkc_valid)
    save_data(os.path.join(setting_dir, f"{dataset_name}_qdkt_mlkc_test.txt"), mlkc_test)
