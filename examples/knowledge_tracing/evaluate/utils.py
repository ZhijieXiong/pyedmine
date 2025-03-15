import os
import torch

from edmine.utils.data_io import read_json
from edmine.dataset.SequentialKTDataset import *


def get_model_info(model_dir_name):
    model_info = model_dir_name.split("@@")
    model_name, setting_name, train_file_name = model_info[0], model_info[1], model_info[2]
    return model_name, setting_name, train_file_name


def config_dimkt(local_params, global_params, global_objects, setting_name, train_file_name):
    # 读取diff数据
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    dimkt_dir = os.path.join(setting_dir, "DIMKT")
    diff_path = os.path.join(dimkt_dir, train_file_name + "_dimkt_diff.json")
    diff = read_json(diff_path)
    question_difficulty = {}
    concept_difficulty = {}
    for k, v in diff["question_difficulty"].items():
        question_difficulty[int(k)] = v
    for k, v in diff["concept_difficulty"].items():
        concept_difficulty[int(k)] = v
    global_objects["dimkt"] = {
        "question_difficulty": question_difficulty,
        "concept_difficulty": concept_difficulty    
    }
    q2c_diff_table = [0] * local_params["num_concept"]
    for c_id, c_diff_id in concept_difficulty.items():
        q2c_diff_table[c_id] = c_diff_id
    global_objects["dimkt"]["q2c_diff_table"] = torch.LongTensor(q2c_diff_table).to(global_params["device"])


def config_lpkt(global_params, global_objects):
    global_objects["LPKT"] = {}
    global_objects["LPKT"]["q_matrix"] = torch.from_numpy(
        global_objects["dataset"]["q_table"]
    ).float().to(global_params["device"]) + 0.03
    q_matrix = global_objects["LPKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1


def config_lbkt(global_params, global_objects):
    q_gamma = global_params["models_config"]["LBKT"]["q_gamma"]
    global_objects["LBKT"] = {}
    global_objects["LBKT"]["q_matrix"] = torch.from_numpy(
        global_objects["dataset"]["q_table"]
    ).float().to(global_params["device"]) + q_gamma
    q_matrix = global_objects["LBKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1
    
    
def select_dataset(model_name):
    if model_name == "DIMKT":
        return DIMKTDataset
    elif model_name == "LPKT":
        return LPKTDataset
    elif model_name == "LBKT":
        return LBKTDataset
    else:
        return BasicSequentialKTDataset
    