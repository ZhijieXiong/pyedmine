import json
import os
import inspect
import numpy as np

from edmine.env.learning_path_recommendation.KTEnv import DLSequentialKTEnv
from edmine.config.env import config_lpr_env
from edmine.config.basic import config_logger
from edmine.config.train import config_lpr_step_trainer, config_optimizer
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


def config_reinforce(local_params):
    agent_name = "Reinforce"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT),
                      "random_generator": np.random.RandomState(local_params["seed"])}
    config_logger(local_params, global_objects)
    config_lpr_env(local_params, global_params, global_objects, MODELS_DIR)
    global_objects["env_simulator"] = DLSequentialKTEnv(global_params, global_objects)
    global_params["loss_config"] = {}
    config_lpr_step_trainer(local_params, global_params, agent_name)
    config_optimizer(local_params, global_params, "concept_action_model")
    config_optimizer(local_params, global_params, "concept_state_model")
    config_optimizer(local_params, global_params, "question_action_model")
    config_optimizer(local_params, global_params, "question_state_model")
    
    # KT模型的配置也在global_params["models_config"]
    global_params["models_config"]["action_model"] = {
        "num_layer": local_params["num_layer_action_model"],
    }
    global_params["models_config"]["state_model"] = {
        "num_layer": local_params["num_layer_state_model"],
    }
    # 特殊配制
    global_params["trainer_config"]["agent_name"] = agent_name
    global_params["trainer_config"]["gamma"] = local_params["gamma"]
    
    global_params["agents_config"] = {
        agent_name: {
            "max_question_attempt": local_params["max_question_attempt"]
        }
    }

    if local_params["save_model"]:
        setting_name = local_params["setting_name"]
        train_file_name = local_params["train_file_name"]

        global_params["trainer_config"]["save_model_dir_name"] = (
            f"{agent_name}@@{setting_name}@@{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@"
            f"{get_now_time().replace(' ', '@').replace(':', '-')}")
        save_params(global_params, MODELS_DIR, global_objects["logger"])
    config_wandb(local_params, global_params, agent_name)

    return global_params, global_objects
