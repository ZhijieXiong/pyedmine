import json
import os
import inspect
import torch
import numpy as np

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


def config_gikt(local_params):
    model_name = "GIKT"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {}
    config_epoch_trainer(local_params, global_params, model_name)
    config_sequential_kt_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)
    
    setting_dir = global_objects["file_manager"].get_setting_dir(local_params["setting_name"])
    gikt_dir = os.path.join(setting_dir, "GIKT")
    if not os.path.exists(gikt_dir):
        os.mkdir(gikt_dir)
    dataset_name = local_params["dataset_name"]
    question_neighbors_path = os.path.join(gikt_dir, f"gikt_question_neighbors_{dataset_name}.npy")
    concept_neighbors_path = os.path.join(gikt_dir, f"gikt_concept_neighbors_{dataset_name}.npy")
    global_objects["GIKT"] = {
        "question_neighbors": torch.from_numpy(np.load(question_neighbors_path)).to(global_params["device"]),
        "concept_neighbors": torch.from_numpy(np.load(concept_neighbors_path)).to(global_params["device"])
    }

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "dim_emb": local_params["dim_emb"],
            "agg_hops": local_params["agg_hops"],
            "rank_k": local_params["rank_k"],
            "dropout4gru": local_params["dropout4gru"],
            "dropout4gnn": local_params["dropout4gnn"],
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
