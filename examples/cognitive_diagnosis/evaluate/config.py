import json
import os
import inspect
from collections import defaultdict

from utils import *

from edmine.data.FileManager import FileManager
from edmine.config.data import config_q_table
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model
from edmine.utils.log import get_now_time
from edmine.utils.data_io import read_cd_file, read_json, write_json


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def config_dlcd(local_params):
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    if local_params.get("save_log", False):
        log_path = os.path.join(MODEL_DIR, local_params["model_dir_name"],
                                f"evaluate_log@{get_now_time().replace(' ', '@').replace(':', '-')}.txt")
    else:
        log_path = None
    config_logger(local_params, global_objects, log_path)
    config_general_dl_model(local_params, global_params)
    config_q_table(local_params, global_params, global_objects)

    model_name, setting_name, train_file_name = get_model_info(local_params["model_dir_name"])
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    data_statics_path = os.path.join(setting_dir, f"{local_params['dataset_name']}_statics.txt")
    with open(data_statics_path, "r") as f:
        s = f.readline()
        local_params["num_user"] = int(s.split(":")[1].strip())
    if model_name == "RCD":
        config_rcd(local_params, global_params, global_objects, setting_dir, train_file_name)
    elif model_name == "HyperCD":
        config_hyper_cd(local_params, global_params, global_objects, setting_dir, train_file_name)
        
    # 配置cold start
    global_params["dlcd"] = {
        "question_cold_start": local_params["question_cold_start"],
        "user_cold_start": local_params["user_cold_start"],
    }
    cold_start_dir = os.path.join(setting_dir, "data4cold_start")
    if not os.path.exists(cold_start_dir):
        os.mkdir(cold_start_dir)
    question_cold_start = global_params["question_cold_start"]
    user_cold_start = global_params["user_cold_start"]
    if (question_cold_start >= 0) or (user_cold_start >= 1):
        train_file_path = os.path.join(setting_dir, train_file_name + ".txt")
        train_data = read_cd_file(train_file_path)
        cold_start_question_path = os.path.join(cold_start_dir, f"cold_start_question_start_{question_cold_start}.json")
        cold_start_user_path = os.path.join(cold_start_dir, f"cold_start_user_{user_cold_start}.json")
        
        if os.path.exists(cold_start_question_path):
            global_objects["cold_start_question"] = read_json(cold_start_question_path)
        else:
            num_q_in_train = defaultdict(int)
            for interaction in train_data:
                question_id = interaction["question_id"]
                num_q_in_train[question_id] += 1
            global_objects["num_q_in_train"] = num_q_in_train
            global_objects["cold_start_question"] = []
            for question_id, num_question in num_q_in_train.items():
                if num_question <= local_params["question_cold_start"]:
                    global_objects["cold_start_question"].append(question_id)
            write_json(global_objects["cold_start_question"], cold_start_question_path)
                    
        if os.path.exists(cold_start_user_path):
            global_objects["cold_start_user"] = read_json(cold_start_user_path)
        else:
            num_u_in_train = defaultdict(int)
            for interaction in train_data:
                user_id = interaction["user_id"]
                num_u_in_train[user_id] += 1
            global_objects["num_u_in_train"] = num_u_in_train
            global_objects["cold_start_user"] = []
            for user_id, num_user in num_u_in_train.items():
                if num_user <= local_params["user_cold_start"]:
                    global_objects["cold_start_user"].append(user_id)
            write_json(global_objects["cold_start_user"], cold_start_user_path)
        
    model_dir = os.path.join(MODEL_DIR, local_params["model_dir_name"])
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_name"], local_params["model_name_in_ckt"])
    global_params["evaluator_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}

    return global_params, global_objects
