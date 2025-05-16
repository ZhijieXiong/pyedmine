import argparse
import numpy as np

import config

from edmine.data.FileManager import FileManager
from edmine.dataset.split_seq import truncate2multi_seq
from edmine.dataset.split_dataset import n_fold_split
from edmine.utils.data_io import write_kt_file, read_kt_file
from edmine.utils.parse import get_kt_data_statics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)

    setting = {
        "name": "pykt_setting",
        "max_seq_len": 200,
        "min_seq_len": 2,
        "n_fold": 5,
        "test_radio": 0.2,
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(setting["name"], setting)
    data = read_kt_file(file_manager.get_preprocessed_path(params["dataset_name"]))
    if params["dataset_name"] in ["junyi2015", "edi2020-task1"]:
        # 只取长度最长的5000条序列
        seq_lens = list(map(lambda x: x["seq_len"], data))
        max_indices = np.argpartition(np.array(seq_lens), -5000)[-5000:]
        data_ = []
        for i in max_indices:
            data_.append(data[i])
        data = data_
    q_table = file_manager.get_q_table(params["dataset_name"])
    data_statics = get_kt_data_statics(data, q_table)
    print(f"data statics: {data_statics}")
        
    dataset_truncated = truncate2multi_seq(data,
                                           setting["min_seq_len"],
                                           setting["max_seq_len"],)
    n_fold_split(params["dataset_name"], dataset_truncated, setting, file_manager, write_kt_file, "kt")
