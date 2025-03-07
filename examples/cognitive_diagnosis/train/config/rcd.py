import json
import os
import dgl
import inspect

from edmine.config.data import config_q_table, config_cd_dataset
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


def build_graph(g_path, node, directed=True):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []
    with open(g_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))
    if directed:
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    else:
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        return g
    

def config_rcd(local_params):
    model_name = "RCD"

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
            "embed_config": {
                "user": {
                    "num_item": local_params["num_user"],
                    "dim_item": local_params["num_concept"],
                    "init_method": "xavier_normal"
                },
                "question": {
                    "num_item": local_params["num_question"],
                    "dim_item": local_params["num_concept"],
                    "init_method": "xavier_normal"
                },
                "concept": {
                    "num_item": local_params["num_concept"],
                    "dim_item": local_params["num_concept"],
                    "init_method": "xavier_normal"
                }
            }
        }
    }
    
    # 加载需要的数据
    setting_dir = global_objects["file_manager"].get_setting_dir(local_params["setting_name"])
    graph_dir = os.path.join(setting_dir, "RCD")
    dataset_name = local_params["dataset_name"]
    train_file_name = local_params["train_file_name"]
    global_objects["dataset"]["local_map"] = {
        'directed_g': build_graph(
            os.path.join(graph_dir, f"{dataset_name}_K_Directed.txt"), 
            local_params["num_concept"],
            True
        ).to(global_params["device"]),
        'undirected_g': build_graph(
            os.path.join(graph_dir, f"{dataset_name}_K_Undirected.txt"), 
            local_params["num_concept"],
            False
        ).to(global_params["device"]),
        'k_from_e': build_graph(
            os.path.join(graph_dir, f"k_from_e_{train_file_name}"), 
            local_params["num_concept"] + local_params["num_question"],
            True
        ).to(global_params["device"]),
        'e_from_k': build_graph(
            os.path.join(graph_dir, f"e_from_k_{train_file_name}"), 
            local_params["num_concept"] + local_params["num_question"],
            True
        ).to(global_params["device"]),
        'u_from_e': build_graph(
            os.path.join(graph_dir, f"u_from_e_{train_file_name}"), 
            local_params["num_question"] + local_params["num_user"],
            True
        ).to(global_params["device"]),
        'e_from_u': build_graph(
            os.path.join(graph_dir, f"e_from_u_{train_file_name}"), 
            local_params["num_question"] + local_params["num_user"],
            True
        ).to(global_params["device"])
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
