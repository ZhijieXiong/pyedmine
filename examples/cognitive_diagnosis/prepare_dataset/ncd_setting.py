import argparse

import config

from edmine.data.FileManager import FileManager
from edmine.dataset.split_dataset import n_fold_split
from edmine.utils.data_io import read_kt_file, write_cd_file
from edmine.utils.parse import kt_data2cd_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)

    setting = {
        "name": "ncd_setting",
        "n_fold": 5,
        "test_radio": 0.2,
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(setting["name"], setting)
    kt_data_ = read_kt_file(file_manager.get_preprocessed_path(params["dataset_name"]))
    if "SLP" in params["dataset_name"]:
        kt_data = []
        for user_data in kt_data_:
            term_data = {
                k: v if type(v) is not list else [] for k, v in user_data.items()
            }
            mode_seq = user_data["mode_seq"]
            for i, m in enumerate(mode_seq):
                if m == 0:
                    continue
                for k, v in user_data.items():
                    if type(v) is list:
                        term_data[k].append(user_data[k][i])
            term_data["seq_len"] = len(term_data["correctness_seq"])
            if term_data["seq_len"] > 1:
                kt_data.append(term_data)
    else:
        kt_data = kt_data_
    cd_data = kt_data2cd_data(kt_data)
    n_fold_split(params["dataset_name"], cd_data, setting, file_manager, write_cd_file, "cd")
