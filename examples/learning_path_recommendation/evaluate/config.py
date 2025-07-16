import json
import os
import inspect
import numpy as np

from edmine.data.FileManager import FileManager
from edmine.env.learning_path_recommendation.KTEnv import DLSequentialKTEnv
from edmine.agent.learning_path_recommendation.RandomSingleGoalAgent import RandomSingleGoalAgent
from edmine.utils.log import get_now_time
from edmine.config.basic import config_logger
from edmine.config.env import config_lpr_env


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def config_lpr(local_params):
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    if local_params.get("save_log", False):
        log_path = os.path.join(MODEL_DIR, local_params["agent_dir_name"],
                                f"evaluate_log@{get_now_time().replace(' ', '@').replace(':', '-')}.txt")
    else:
        log_path = None
    if local_params.get("save_all_sample", False):
        all_sample_path = os.path.join(MODEL_DIR, local_params["agent_dir_name"],
                                       f"all_sample_evaluation.txt")
    else:
        all_sample_path = None
    global_params["all_sample_path"] = all_sample_path
    config_logger(local_params, global_objects, log_path)
    config_lpr_env(local_params, global_params, global_objects, MODEL_DIR)
    global_objects["env_simulator"] = DLSequentialKTEnv(global_params, global_objects)
    if "RandomAgent" in local_params["agent_dir_name"]:
        config_random_agent(local_params, global_params, global_objects)
    else:
        pass
    setting_name = local_params["setting_name"]
    test_file_name = local_params["test_file_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    global_params["datasets_config"] = {
        "test": {
            "file_path": os.path.join(setting_dir, test_file_name),
            "batch_size": local_params["batch_size"]
        }
    }
    global_params["evaluate_config"] = {
        "master_threshold": local_params["master_threshold"]
    }
    
    return global_params, global_objects


def config_random_agent(local_params, global_params, global_objects):
    global_objects["agent_class"] = RandomSingleGoalAgent
    agent_name = "RandomAgent"
    global_objects["random_generator"] = np.random.RandomState(local_params["seed"])
    _, concept_rec_strategy, max_attempt_per_concept = local_params["agent_dir_name"].split("@@")
    global_params["agents_config"] = {
        agent_name: {
            "num_question": global_objects["dataset"]["q_table"].shape[0],
            "num_concept": global_objects["dataset"]["q_table"].shape[1],
            "concept_rec_strategy": concept_rec_strategy,
            "max_attempt_per_concept": int(max_attempt_per_concept)
        }
    }
