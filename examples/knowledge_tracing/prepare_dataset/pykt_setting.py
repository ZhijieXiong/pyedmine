import argparse

import config

from edmine.data.FileManager import FileManager
from edmine.dataset.split_seq import truncate2multi_seq
from edmine.dataset.split_dataset import n_fold_split
from edmine.utils.data_io import write_kt_file, read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ednet-kt1")
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
    dataset_truncated = truncate2multi_seq(read_kt_file(file_manager.get_preprocessed_path(params["dataset_name"])),
                                           setting["min_seq_len"],
                                           setting["max_seq_len"],)
    n_fold_split(params["dataset_name"], dataset_truncated, setting, file_manager, write_kt_file, "kt")
