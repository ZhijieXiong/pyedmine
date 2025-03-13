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
from edmine.utils.data_io import save_params

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODELS_DIR = settings["MODELS_DIR"]


def config_lbkt(local_params):
    model_name = "LBKT"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {}
    config_epoch_trainer(local_params, global_params, model_name)
    config_sequential_kt_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)
    
    global_objects["LBKT"] = {}
    global_objects["LBKT"]["q_matrix"] = torch.from_numpy(
        global_objects["dataset"]["q_table"]
    ).float().to(global_params["device"]) + local_params["q_gamma"]
    q_matrix = global_objects["LBKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "embed_config": {
                "question": {
                    "num_item": local_params["num_question"],
                    "dim_item": local_params["dim_question"],
                    "init_method": "xavier_normal"
                },
                "correctness": {
                    "num_item": 2,
                    "dim_item": local_params["dim_correctness"],
                    "init_method": "xavier_normal"
                }
            },
            "dim_h": local_params["dim_h"],
            "dim_factor": local_params["dim_factor"],
            "r": local_params["r"],
            "d": local_params["d"],
            "k": local_params["k"],
            "b": local_params["b"],
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
