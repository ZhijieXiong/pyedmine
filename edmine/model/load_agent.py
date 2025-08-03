import os
import torch

from edmine.utils.parse import str_dict2params
from edmine.utils.data_io import read_json
from edmine.model.registry import MODEL_REGISTRY
from edmine.model.utils import import_all_models


def load_lpr_agent(global_params, global_objects, save_agent_dir, ckt_name="saved.ckt"):
    # 自动导入所有模型模块（必须放在使用注册表之前）
    import_all_models()
    
    params_path = os.path.join(save_agent_dir, "params.json")
    saved_params = read_json(params_path)
    global_params["models_config"] = str_dict2params(saved_params["models_config"])
    global_params["agents_config"] = str_dict2params(saved_params["agents_config"])
    
    ckt_path = os.path.join(save_agent_dir, ckt_name)
    agent_name = os.path.basename(save_agent_dir).split("@@")[0]
    agent_class = MODEL_REGISTRY[agent_name]
    agent = agent_class(global_params, global_objects)
    if global_params["device"] == "cpu":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('cpu'), weights_only=True)
    elif global_params["device"] == "mps":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('mps'), weights_only=True)
    else:
        saved_ckt = torch.load(ckt_path, weights_only=True)
    for model_name, model in global_objects["lpr_models"].items():
        model.load_state_dict(saved_ckt[model_name])

    return agent
