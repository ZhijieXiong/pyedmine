import os
import torch

from edmine.config.data import config_q_table
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model
from edmine.utils.data_io import read_json


def config_lpr_env(local_params, global_params, global_objects, model_dir):
    global_params["kt_model_config"] = {
        "seq_data_keys": ["question_seq", "correctness_seq", "mask_seq"],
        "id_data_keys": ["seq_len"]
    }
    config_general_dl_model(local_params, global_params)
    if local_params.get("dataset_name", False):
        config_q_table(local_params, global_params, global_objects)
    model_dir_name = local_params["model_dir_name"]
    model_name, setting_name, train_file_name = get_model_info(local_params["model_dir_name"])
    if model_dir_name.startswith("DIMKT@@"):
        setting_name = local_params["kt_setting_name"]
        config_dimkt(local_params, global_params, global_objects, setting_name, train_file_name)
    model_dir = os.path.join(model_dir, model_dir_name)
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_file_name"], local_params["model_name_in_ckt"])
    global_params["env_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}
    

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
