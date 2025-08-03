import os
import torch

from edmine.utils.parse import str_dict2params
from edmine.utils.data_io import read_json
from edmine.model.registry import MODEL_REGISTRY
from edmine.model.utils import import_all_models


def load_dl_model(global_params, global_objects, save_model_dir, ckt_name="saved.ckt", model_name_in_ckt="best_valid"):
    # 自动导入所有模型模块（必须放在使用注册表之前）
    import_all_models()
    
    params_path = os.path.join(save_model_dir, "params.json")
    saved_params = read_json(params_path)
    global_params["models_config"] = str_dict2params(saved_params["models_config"])

    ckt_path = os.path.join(save_model_dir, ckt_name)
    model_name = os.path.basename(save_model_dir).split("@@")[0]
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(global_params, global_objects).to(global_params["device"])
    if global_params["device"] == "cpu":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('cpu'), weights_only=True)
    elif global_params["device"] == "mps":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('mps'), weights_only=True)
    else:
        saved_ckt = torch.load(ckt_path, weights_only=True)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    return model
