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


def config_hdlpkt(local_params):
    model_name = "HDLPKT"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {}
    config_epoch_trainer(local_params, global_params, model_name)
    config_sequential_kt_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)
    
    global_objects["HDLPKT"] = {}
    global_objects["HDLPKT"]["q_matrix"] = torch.from_numpy(
        global_objects["dataset"]["q_table"]
    ).float().to(global_params["device"]) + 0.03
    q_matrix = global_objects["HDLPKT"]["q_matrix"]
    q_matrix[q_matrix > 1] = 1

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "embed_config": {
                "user": {
                    "num_item": 100000,
                    "dim_item": local_params["dim_k"],
                    "init_method": "xavier_normal"
                },
                "question": {
                    "num_item": local_params["num_question"],
                    "dim_item": local_params["dim_e"],
                    "init_method": "xavier_normal"
                },
                "interaction": {
                    "num_item": local_params["num_concept"] * 2,
                    "dim_item": local_params["dim_k"],
                    "init_method": "xavier_normal"
                },
                "use_time": {
                    "num_item": 60 * 60 + 1,
                    "dim_item": local_params["dim_k"],
                    "init_method": "xavier_normal"
                },
                "interval_time": {
                    "num_item": 60 * 24 * 30 + 1,
                    "dim_item": local_params["dim_k"],
                    "init_method": "xavier_normal"
                },
                "correctness": {
                    "init_method": "init_correctness_1",
                    "dim_item": local_params["dim_k"]
                }
            },
            "dim_e": local_params["dim_e"],
            "dim_a": local_params["dim_a"],
            "dim_k": local_params["dim_k"],
            "dropout": local_params["dropout"],
            "max_seq_length": local_params["max_seq_length"],
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
