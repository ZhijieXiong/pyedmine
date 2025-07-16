import os
import torch

from edmine.config.data import config_q_table
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model


def config_lpr_env(local_params, global_params, global_objects, model_dir):
    config_general_dl_model(local_params, global_params)
    if local_params.get("dataset_name", False):
        config_q_table(local_params, global_params, global_objects)
    model_name = local_params["model_dir_name"]
    if model_name.startswith("ABQR@@"):
        setting_name = local_params["kt_setting_name"]
        config_abqr(local_params, global_params, global_objects, setting_name)
    model_dir = os.path.join(model_dir, model_name)
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_name"], local_params["model_name_in_ckt"])
    model.eval()
    global_params["env_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}
    

def config_abqr(local_params, global_params, global_objects, setting_name):
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    abqr_dir = os.path.join(setting_dir, "ABQR")
    dataset_name = local_params["dataset_name"]
    if dataset_name in ["assist2009", "moocradar-C746997", "ednet-kt1", "xes3g5m"]:
        graph_path = os.path.join(abqr_dir, f"abqr_graph_{dataset_name}-single-concept.pt")
    else:
        graph_path = os.path.join(abqr_dir, f"abqr_graph_{dataset_name}.pt")
    global_objects["ABQR"] = {
        "gcn_adj": torch.load(graph_path, weights_only=True).to(global_params["device"])
    }