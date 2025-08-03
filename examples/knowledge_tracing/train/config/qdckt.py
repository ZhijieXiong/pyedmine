import json
import os
import inspect
import torch
import math
from copy import deepcopy

from edmine.config.data import config_q_table, config_sequential_kt_dataset
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.config.train import config_epoch_trainer, config_optimizer
from edmine.config.train import config_wandb
from edmine.data.FileManager import FileManager
from edmine.utils.log import get_now_time
from edmine.utils.data_io import save_params, read_json

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODELS_DIR = settings["MODELS_DIR"]


def config_qdckt(local_params):
    model_name = "QDCKT"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {
        "qdckt loss": local_params["w_qdckt_loss"]
    }
    config_epoch_trainer(local_params, global_params, model_name)
    config_sequential_kt_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)
    
    # 读取diff数据
    setting_dir = global_objects["file_manager"].get_setting_dir(local_params["setting_name"])
    dimkt_dir = os.path.join(setting_dir, "QDCKT")
    diff_path = os.path.join(dimkt_dir, local_params["train_file_name"].replace(".txt", "_qdckt_diff.json"))
    diff = read_json(diff_path)
    question_difficulty = {}
    for k, v in diff["question_difficulty"].items():
        question_difficulty[int(k)] = v
    num_que_diff = diff["num_question_diff"]
    global_objects[model_name] = {
        "question_difficulty": question_difficulty,
        "num_question_diff": num_que_diff
    }
    w_size = local_params["window_size"]
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
    global_objects[model_name]["q2diff_transfer_table"] = torch.LongTensor(q2diff_transfer_table).to(global_params["device"])
    q2diff_weight_table = torch.FloatTensor(q2diff_weight_table).to(global_params["device"])
    # 按照论文中所说归一化
    global_objects[model_name]["q2diff_weight_table"] = q2diff_weight_table / q2diff_weight_table.sum(dim=1, keepdim=True)

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "embed_config": {
                "concept": {
                    "num_item": local_params["num_concept"],
                    "dim_item": local_params["dim_emb"]
                },
                "question_diff": {
                    "num_item": num_que_diff,
                    "dim_item": local_params["dim_emb"]
                },
                "correctness": {
                    "num_item": 2,
                    "dim_item": local_params["dim_correctness"]
                },
            },
            "dropout": local_params["dropout"],
            "dim_latent": local_params["dim_latent"],
            "rnn_type": local_params["rnn_type"],
            "num_rnn_layer": local_params["num_rnn_layer"],
            "window_size": local_params["window_size"],
            "predictor_config": {
                "type": "direct",
                "dropout": local_params["dropout"],
                "num_predict_layer": local_params["num_predict_layer"],
                "dim_predict_in": local_params["dim_latent"] + local_params["dim_correctness"],
                "dim_predict_mid": local_params["dim_predict_mid"],
                "dim_predict_out": 1,
                "activate_type": local_params["activate_type"],
                "last_layer_max_value": 1
            }
        }
    }

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["trainer_config"]["save_model_dir_name"] = (
            f"{model_name}@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")
        save_params(global_params, MODELS_DIR, global_objects["logger"])
    config_wandb(local_params, global_params, model_name)

    return global_params, global_objects
