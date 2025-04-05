import os
import torch

from edmine.utils.data_io import read_json
from edmine.dataset.SequentialKTDataset import *
from edmine.dataset.SequentialKTDatasetWithSample import *


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
    

def config_hdlpkt(global_params, global_objects):
    global_objects["HDLPKT"] = {}
    global_objects["HDLPKT"]["q_matrix"] = torch.from_numpy(
        global_objects["dataset"]["q_table"]
    ).float().to(global_params["device"]) + 0.03
    q_matrix = global_objects["HDLPKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1


def config_lbkt(global_params, global_objects):
    q_gamma = global_params["models_config"]["LBKT"]["q_gamma"]
    global_objects["LBKT"] = {}
    global_objects["LBKT"]["q_matrix"] = torch.from_numpy(
        global_objects["dataset"]["q_table"]
    ).float().to(global_params["device"]) + q_gamma
    q_matrix = global_objects["LBKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1
    
    
def config_qdckt(global_params, global_objects, setting_name, train_file_name):
    # 读取diff数据
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    dimkt_dir = os.path.join(setting_dir, "QDCKT")
    diff_path = os.path.join(dimkt_dir, train_file_name + "_qdckt_diff.json")
    diff = read_json(diff_path)
    question_difficulty = {}
    for k, v in diff["question_difficulty"].items():
        question_difficulty[int(k)] = v
    num_que_diff = diff["num_question_diff"]
    global_objects["qdckt"] = {
        "question_difficulty": question_difficulty,
        "num_question_diff": num_que_diff
    }
    w_size = global_params["models_config"]["QDCKT"]["window_size"]
    assert (w_size % 2 == 1) and (w_size >= 1), "window_size must an odd number greater than or equal to 1"
    q2diff_transfer_table = []
    q2diff_weight_table = []
    ws = [0.5 ** abs(i - w_size//2) for i in range(w_size)]
    for q_diff in range(num_que_diff):
        if q_diff < (w_size / 2):
            q2diff_transfer_table.append(list(range(w_size//2 + 1 + q_diff)) + [0] * (w_size // 2 - q_diff))
            q2diff_weight_table.append(ws[-(w_size//2+1 + q_diff):] + [0] * (w_size // 2 - q_diff))
        elif (num_que_diff - q_diff) < (w_size / 2):
            q2diff_transfer_table.append(list(range(q_diff - w_size//2, num_que_diff)) + [0] * (w_size // 2 - (num_que_diff - q_diff) + 1))
            q2diff_weight_table.append(ws[:-(w_size//2+1-(num_que_diff - q_diff))] + [0] * (w_size // 2 - (num_que_diff - q_diff) + 1))
        else:
            q2diff_transfer_table.append(list(range(q_diff-w_size//2, q_diff-w_size//2 + w_size)))
            q2diff_weight_table.append(deepcopy(ws))
    global_objects["qdckt"]["q2diff_transfer_table"] = torch.LongTensor(q2diff_transfer_table).to(global_params["device"])
    q2diff_weight_table = torch.FloatTensor(q2diff_weight_table).to(global_params["device"])
    # 按照论文中所说归一化
    global_objects["qdckt"]["q2diff_weight_table"] = q2diff_weight_table / q2diff_weight_table.sum(dim=1, keepdim=True)


def config_grkt(global_objects, setting_name, train_file_name):
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    grkt_dir = os.path.join(setting_dir, "GRKT")
    rel_map_path = os.path.join(grkt_dir, train_file_name + "_grkt_rel_map.npy")
    pre_map_path = os.path.join(grkt_dir, train_file_name + "_grkt_pre_map.npy")
    global_objects["GRKT"] = {
        "rel_map": np.load(rel_map_path),
        "pre_map": np.load(pre_map_path)
    }
    

def config_abqr(local_params, global_params, global_objects, setting_name):
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    abqr_dir = os.path.join(setting_dir, "ABQR")
    dataset_name = local_params["dataset_name"]
    graph_path = os.path.join(abqr_dir, f"abqr_graph_{dataset_name}.pt")
    global_objects["ABQR"] = {
        "gcn_adj": torch.load(graph_path).to(global_params["device"])
    }
    
    
def select_dataset(model_name):
    if model_name == "DIMKT":
        return DIMKTDataset
    elif model_name == "LPKT":
        return LPKTDataset
    elif model_name == "LBKT":
        return LBKTDataset
    elif model_name == "DKTForget":
        return DKTForgetDataset
    elif model_name == "QDCKT":
        return QDCKTDataset
    elif model_name == "ATDKT":
        return ATDKTDataset
    elif model_name == "CLKT":
        return CLKTDataset
    elif model_name == "DTransformer":
        return DTransformerDataset
    elif model_name == "GRKT":
        return GRKTDataset
    elif model_name == "CKT":
        return CKTDataset
    elif model_name == "HDLPKT":
        return HDLPKTDataset
    else:
        return BasicSequentialKTDataset
    