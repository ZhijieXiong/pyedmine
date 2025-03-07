import json
import os
import inspect
import numpy as np

from edmine.config.data import config_q_table, config_cd_dataset
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.config.train import config_epoch_trainer, config_optimizer
from edmine.config.train import config_wandb
from edmine.data.FileManager import FileManager
from edmine.utils.log import get_now_time
from edmine.utils.data_io import save_params
from edmine.model.module.Graph import HyperCDgraph

current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODELS_DIR = settings["MODELS_DIR"]
    

def config_hyper_cd(local_params):
    model_name = "HyperCD"

    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    config_logger(local_params, global_objects)
    config_general_dl_model(local_params, global_params)
    global_params["loss_config"] = {}
    config_epoch_trainer(local_params, global_params, model_name)
    config_cd_dataset(local_params, global_params, global_objects)
    config_optimizer(local_params, global_params, model_name)
    config_q_table(local_params, global_params, global_objects)

    # 模型参数
    global_params["models_config"] = {
        model_name: {
            "num_concept": local_params["num_concept"],
            "num_question": local_params["num_question"],
            "num_user": local_params["num_user"],
            "num_layer": local_params["num_layer"],
            "dim_feature": local_params["dim_feature"],
            "dim_emb": local_params["dim_emb"],
            "leaky": local_params["leaky"],
        }
    }
    
    # 加载需要的数据
    file_manager = global_objects["file_manager"]
    setting_dir = file_manager.get_setting_dir(local_params["setting_name"])
    graph_dir = os.path.join(setting_dir, "HyperCD")
    dataset_name = local_params["dataset_name"]
    train_file_name = local_params["train_file_name"]
    
    q_table = file_manager.get_q_table(dataset_name)
    Hu_path = os.path.join(graph_dir, f"user_hyper_graph_{train_file_name.replace('txt', 'npy')}")
    Hu = HyperCDgraph(np.load(Hu_path))
    Hq = q_table.copy()
    Hc = q_table.T.copy()
    Hq = HyperCDgraph(Hq[:, np.count_nonzero(Hq, axis=0) >= 2])
    Hc = HyperCDgraph(Hc[:, np.count_nonzero(Hc, axis=0) >= 2])
    global_objects["dataset"]["hyper_graph"] = {
        "question": Hq,
        "concept": Hc,
        "user": Hu
    }
    global_objects["dataset"]["adj"] = {
        "question": Hq.to_tensor_nadj().to(global_params["device"]),
        "concept": Hc.to_tensor_nadj().to(global_params["device"]),
        "user": Hu.to_tensor_nadj().to(global_params["device"])
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
