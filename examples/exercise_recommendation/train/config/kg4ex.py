import json
import os
import inspect

from edmine.config.data import config_q_table, config_kg4ex_dataset
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.config.train import config_exercise_recommendation_trainer, config_optimizer
from edmine.config.train import config_wandb
from edmine.data.FileManager import FileManager
from edmine.utils.log import get_now_time
from edmine.utils.data_io import save_params, read_id_map_kg4ex

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODELS_DIR = settings["MODELS_DIR"]


def config_kg4ex(local_params):
    model_name = "KG4EX"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {
        "regularization loss": local_params["w_reg_loss"]
    }
    config_exercise_recommendation_trainer(local_params, global_params, model_name)
    global_params["trainer_config"]["evaluate_batch_size"] = local_params["evaluate_batch_size"]
    config_kg4ex_dataset(local_params, global_params)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)
    # 读取id map
    setting_name = local_params["setting_name"]
    dataset_name = local_params["dataset_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    kg4ex_dir = os.path.join(setting_dir, "kg4ex")
    global_objects["dataset"]["entity2id"] = read_id_map_kg4ex(os.path.join(kg4ex_dir, f'{dataset_name}_entities_kg4ex.dict'))
    # 存储relations
    relations_path = os.path.join(kg4ex_dir, "relations_kg4ex.dict")
    if not os.path.exists(relations_path):
        scores = [round(i * 0.01, 2) for i in range(101)]
        with open(relations_path, "w") as fs:
            for i, s in enumerate(scores):
                fs.write(f"{i}\tmlkc{s}\n")
            for i, s in enumerate(scores):
                fs.write(f"{i + 101}\tpkc{s}\n")
            for i, s in enumerate(scores):
                fs.write(f"{i + 202}\tefr{s}\n")
            fs.write("303\trec")
    global_objects["dataset"]["relation2id"] = read_id_map_kg4ex(os.path.join(kg4ex_dir, 'relations_kg4ex.dict'))

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "dim": local_params["dim"],
            "negative_sample_size": local_params["negative_sample_size"],
            "model_selection": local_params["model_selection"],
            "gamma": local_params["gamma"],
            "double_entity_embedding": local_params["double_entity_embedding"],
            "double_relation_embedding": local_params["double_relation_embedding"],
            "negative_adversarial_sampling": local_params["negative_adversarial_sampling"],
            "uni_weight": local_params["uni_weight"],
            "adversarial_temperature": local_params["adversarial_temperature"],
            "epsilon": local_params["epsilon"],
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
