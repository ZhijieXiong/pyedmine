import json
import os
import inspect

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


def config_akt_core(local_params):
    model_name = "AKT_CORE"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {
        "rasch loss": local_params["w_rasch_loss"]
    }
    config_epoch_trainer(local_params, global_params, model_name)
    config_sequential_kt_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)

    # 模型参数
    global_params["models_config"] = {
        # 因为AKT_CORE要复用AKT的代码
        "AKT": {
            "embed_config": {
                "question_diff": {
                    "num_item": local_params["num_question"],
                    "dim_item": 1,
                    "init_method": "init_zero"
                },
                "concept": {
                    "num_item": local_params["num_concept"],
                    "dim_item": local_params["dim_model"]
                },
                "concept_var": {
                    "num_item": local_params["num_concept"],
                    "dim_item": local_params["dim_model"]
                },
                "interaction": {
                    "num_item": (local_params["num_concept"] * 2) if local_params["separate_qa"] else 2,
                    "dim_item": local_params["dim_model"]
                },
                "interaction_var": {
                    "num_item": 2,
                    "dim_item": local_params["dim_model"]
                }
            },
            "fusion_mode": local_params["fusion_mode"],
            "dim_model": local_params["dim_model"],
            "num_block": local_params["num_block"],
            "num_head": local_params["num_head"],
            "dim_ff": local_params["dim_ff"],
            "seq_len": local_params["seq_len"],
            "dropout": local_params["dropout"],
            "key_query_same": local_params["key_query_same"],
            "separate_qa": local_params["separate_qa"]
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
