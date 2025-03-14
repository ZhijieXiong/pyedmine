import json
import os
import inspect

from utils import *

from edmine.data.FileManager import FileManager
from edmine.config.data import config_q_table
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model
from edmine.utils.log import get_now_time


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def config_dler(local_params):
    model_name, setting_name, train_file_name = get_model_info(local_params["model_dir_name"])
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    if local_params.get("save_log", False):
        log_path = os.path.join(MODEL_DIR, local_params["model_dir_name"],
                                f"evaluate_log@{get_now_time().replace(' ', '@').replace(':', '-')}.txt")
    else:
        log_path = None
    config_logger(local_params, global_objects, log_path)
    config_general_dl_model(local_params, global_params)
    global_params["dler"] = {
        "kg4ex": {
            "batch_size": local_params.get("evaluate_batch_size", 1)
        },
        "top_ns": eval(local_params["top_ns"])
    }
    config_q_table(local_params, global_params, global_objects)
    if model_name == "KG4EX":
        config_kg4ex(local_params, global_objects, setting_name)
    model_dir = os.path.join(MODEL_DIR, local_params["model_dir_name"])
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_name"], local_params["model_name_in_ckt"])
    global_params["evaluator_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}

    return global_params, global_objects
