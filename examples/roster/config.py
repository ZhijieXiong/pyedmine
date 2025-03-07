import json
import os
import inspect

from edmine.data.FileManager import FileManager
from edmine.config.data import config_q_table
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def config_roster(local_params):
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_general_dl_model(local_params, global_params)

    config_q_table(local_params, global_params, global_objects)
    model_name = local_params["model_dir_name"]
    model_dir = os.path.join(MODEL_DIR, model_name)
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_name"], local_params["model_name_in_ckt"])
    model.eval()
    global_params["roster_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}

    return global_params, global_objects
