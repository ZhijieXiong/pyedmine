import torch

from edmine.utils.use_torch import is_cuda_available, is_mps_available


def config_general_dl_model(local_params, global_params):
    if is_cuda_available() and not local_params.get("use_cpu", False):
        device = "cuda"
    elif is_mps_available() and not local_params.get("use_cpu", False):
        device = "mps"
    else:
        device = "cpu"
    global_params["device"] = device
    if local_params.get("debug_mode", False):
        torch.autograd.set_detect_anomaly(True)
    global_params["seed"] = local_params.get("seed", 0)


