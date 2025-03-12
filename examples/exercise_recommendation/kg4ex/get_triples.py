import argparse
import os

import config
from utils import *

from edmine.utils.data_io import read_mlkc_data
from edmine.utils.parse import q2c_from_q_table
from edmine.data.FileManager import FileManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--kt_model_name", type=str, default="dkt")
    parser.add_argument("--efr_theta", type=float, default=0.2)
    parser.add_argument("--delta1", type=float, default=0.7)
    parser.add_argument("--delta2", type=float, default=0.7)
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    Q_table = file_manager.get_q_table(params["dataset_name"])
    question2concept = q2c_from_q_table(Q_table)

    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    kg4ex_dir = os.path.join(setting_dir, "kg4ex")
    dataset_name = params["dataset_name"]
    kt_model_name = params["kt_model_name"]
    efr_theta = params["efr_theta"]

    mlkc_train = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_{kt_model_name}_mlkc_train.txt"))
    mlkc_valid = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_{kt_model_name}_mlkc_valid.txt"))
    mlkc_test = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_{kt_model_name}_mlkc_test.txt"))

    pkc_train = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_pkc_train.txt"))
    pkc_valid = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_pkc_valid.txt"))
    pkc_test = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_pkc_test.txt"))

    efr_train = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_efr_{efr_theta}_train.txt"))
    efr_valid = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_efr_{efr_theta}_valid.txt"))
    efr_test = read_mlkc_data(os.path.join(kg4ex_dir, f"{dataset_name}_efr_{efr_theta}_test.txt"))

    rec_ex_train = {}
    train_user_ids = mlkc_train.keys()
    for train_user_id in train_user_ids:
        mlkc, pkc, efr = mlkc_train[train_user_id], pkc_train[train_user_id], efr_train[train_user_id]
        rec_ex_train[train_user_id] = get_recommended_exercises(
            question2concept, Q_table, mlkc, pkc, efr, params["delta1"], params["delta2"], params["top_n"]
        )

    triples_train_path = os.path.join(kg4ex_dir, f"{dataset_name}_train_triples_{kt_model_name}_{efr_theta}.txt")
    triples_valid_path = os.path.join(kg4ex_dir, f"{dataset_name}_valid_triples_{kt_model_name}_{efr_theta}.txt")
    triples_test_path = os.path.join(kg4ex_dir, f"{dataset_name}_test_triples_{kt_model_name}_{efr_theta}.txt")

    save_triples(triples_train_path, mlkc_train, pkc_train, efr_train, rec_ex_train)
    save_triples(triples_valid_path, mlkc_valid, pkc_valid, efr_valid)
    save_triples(triples_test_path, mlkc_test, pkc_test, efr_test)

    # 存储entities
    num_user = len(mlkc_train)
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]
    with open(os.path.join(kg4ex_dir, f"{dataset_name}_entities_kg4ex.dict"), "w") as fs:
        for i, user_id in enumerate(train_user_ids):
            fs.write(f"{i}\tuid{user_id}\n")
        for i in range(num_concept):
            fs.write(f"{i+num_user}\tkc{i}\n")
        for i in range(num_question):
            fs.write(f"{i+num_user+num_concept}\tex{i}\n")
