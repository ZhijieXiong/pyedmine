import json
import os
import inspect
import torch

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


def config_dimkt(local_params):
    model_name = "DIMKT"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {}
    config_epoch_trainer(local_params, global_params, model_name)
    config_sequential_kt_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)
    
    # 读取diff数据
    setting_dir = global_objects["file_manager"].get_setting_dir(local_params["setting_name"])
    dimkt_dir = os.path.join(setting_dir, "DIMKT")
    diff_path = os.path.join(dimkt_dir, local_params["train_file_name"].replace(".txt", "_dimkt_diff.json"))
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

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "embed_config": {
                "concept": {
                    "num_item": local_params["num_concept"],
                    "dim_item": local_params["dim_emb"]
                },
                "question": {
                    "num_item": local_params["num_question"],
                    "dim_item": local_params["dim_emb"]
                },
                "concept_diff": {
                    "num_item": max(concept_difficulty.values()) + 1,
                    "dim_item": local_params["dim_emb"]
                },
                "question_diff": {
                    "num_item": max(question_difficulty.values()) + 1,
                    "dim_item": local_params["dim_emb"]
                },
                "correctness": {
                    "num_item": 2,
                    "dim_item": local_params["dim_emb"]
                },
            },
            "dropout": local_params["dropout"],
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
