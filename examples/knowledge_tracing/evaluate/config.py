import json
import os
import inspect

from edmine.data.FileManager import FileManager
from edmine.config.data import config_q_table
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model
from edmine.utils.check import check_kt_seq_start
from edmine.utils.log import get_now_time


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def config_sequential_dlkt(local_params):
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    if local_params.get("save_log", False):
        log_path = os.path.join(MODEL_DIR, local_params["model_dir_name"],
                                f"evaluate_log@{get_now_time().replace(' ', '@').replace(':', '-')}.txt")
    else:
        log_path = None
    config_logger(local_params, global_objects, log_path)
    config_general_dl_model(local_params, global_params)

    check_kt_seq_start(local_params["seq_start"])
    global_params["sequential_dlkt"] = {
        "seq_start": local_params["seq_start"],
        "cold_start": local_params["cold_start"],
        "multi_step": local_params["multi_step"]
    }

    config_q_table(local_params, global_params, global_objects)
    model_name = local_params["model_dir_name"]
    model_dir = os.path.join(MODEL_DIR, model_name)
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_name"], local_params["model_name_in_ckt"])
    global_params["evaluator_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}

    return global_params, global_objects
