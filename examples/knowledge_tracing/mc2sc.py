import json
import argparse
import os
import inspect
import numpy as np

from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def transform_T(T):
    Q, C = T.shape
    q2c = q2c_from_q_table(T)
    
    c_id_new = {}
    qc_map = {}
    for q_id in range(Q):
        c_ids = q2c[q_id]
        if len(c_ids) > 1:
            c_ids_str = "-".join(list(map(str, c_ids)))
            if c_ids_str not in c_id_new:
                c_id_new[c_ids_str] = C + len(c_id_new)
            qc_map[q_id] = c_id_new[c_ids_str]
        else:
            qc_map[q_id] = c_ids[0]
            
    T_new = np.zeros((Q, C + len(c_id_new)), dtype=int)
    for q_id in range(Q):
        T_new[q_id][qc_map[q_id]] = 1
    
    return T_new[:, ~np.all(T_new == 0, axis=0)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    q_table = file_manager.get_q_table(params["dataset_name"])
    if q_table.sum() > q_table.shape[0]:
        q_table_ = transform_T(q_table)
        new_dir_name = params["dataset_name"] + "-single-concept"
        root_dir = file_manager.get_root_dir()
        new_dir = os.path.join(root_dir, "dataset", "dataset_preprocessed", new_dir_name)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        q_table_path = os.path.join(new_dir, "Q_table.npy")
        np.save(q_table_path, q_table_)
    