import os
import numpy as np

from edmine.utils.use_dgl import build_graph4rcd
from edmine.model.module.Graph import HyperCDgraph


def get_model_info(model_dir_name):
    model_info = model_dir_name.split("@@")
    model_name, setting_name, train_file_name = model_info[0], model_info[1], model_info[2]
    return model_name, setting_name, train_file_name


def config_rcd(local_params, global_params, global_objects, setting_dir, train_file_name):
    graph_dir = os.path.join(setting_dir, "RCD")
    dataset_name = local_params["dataset_name"]
    
    global_objects["dataset"]["local_map"] = {
        'directed_g': build_graph4rcd(
            os.path.join(graph_dir, f"{dataset_name}_K_Directed.txt"), 
            local_params["num_concept"],
            True
        ).to(global_params["device"]),
        'undirected_g': build_graph4rcd(
            os.path.join(graph_dir, f"{dataset_name}_K_Undirected.txt"), 
            local_params["num_concept"],
            False
        ).to(global_params["device"]),
        'k_from_e': build_graph4rcd(
            os.path.join(graph_dir, f"k_from_e_{train_file_name}.txt"), 
            local_params["num_concept"] + local_params["num_question"],
            True
        ).to(global_params["device"]),
        'e_from_k': build_graph4rcd(
            os.path.join(graph_dir, f"e_from_k_{train_file_name}.txt"), 
            local_params["num_concept"] + local_params["num_question"],
            True
        ).to(global_params["device"]),
        'u_from_e': build_graph4rcd(
            os.path.join(graph_dir, f"u_from_e_{train_file_name}.txt"), 
            local_params["num_question"] + local_params["num_user"],
            True
        ).to(global_params["device"]),
        'e_from_u': build_graph4rcd(
            os.path.join(graph_dir, f"e_from_u_{train_file_name}.txt"), 
            local_params["num_question"] + local_params["num_user"],
            True
        ).to(global_params["device"])
    }
    
    
def config_hyper_cd(local_params, global_params, global_objects, setting_dir, train_file_name):
    file_manager = global_objects["file_manager"]
    graph_dir = os.path.join(setting_dir, "HyperCD")
    dataset_name = local_params["dataset_name"]
    
    q_table = file_manager.get_q_table(dataset_name)
    Hu_path = os.path.join(graph_dir, f"user_hyper_graph_{train_file_name}.npy")
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
