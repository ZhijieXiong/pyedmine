import argparse
import os
import random

import config

from edmine.data.FileManager import FileManager
from edmine.dataset.split_seq import truncate2one_seq
from edmine.dataset.split_dataset import kt_select_test_data
from edmine.utils.data_io import write_kt_file, read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)

    setting = {
        "name": "sfkt_setting",
        "max_seq_len": 2000,
        "min_seq_len": 2,
        "test_radio": 0.2,
        "valid_radio": 0.3,
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(setting["name"], setting)
    data = read_kt_file(file_manager.get_preprocessed_path(params["dataset_name"]))
    dataset = truncate2one_seq(data, 2, 2000, True, True)
    test_radio = setting["test_radio"]
    valid_radio = setting["valid_radio"]
    dataset_train_valid, dataset_test = kt_select_test_data(dataset, test_radio, False)
    num_valid = int(len(dataset_train_valid) * valid_radio)
    random.shuffle(dataset_train_valid)
    dataset_valid = dataset_train_valid[:num_valid]
    dataset_train = dataset_train_valid[num_valid:]
    
    setting_dir = file_manager.get_setting_dir(setting["name"])
    write_kt_file(dataset_test, os.path.join(setting_dir, f"{params['dataset_name']}_test.txt"))
    write_kt_file(dataset_valid, os.path.join(setting_dir, f"{params['dataset_name']}_valid.txt"))
    write_kt_file(dataset_train, os.path.join(setting_dir, f"{params['dataset_name']}_train.txt"))
    
