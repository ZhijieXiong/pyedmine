import wandb

from edmine.utils.log import get_now_time


def config_epoch_trainer(local_params, global_params, model_name):
    global_params["trainer_config"] = {
        "batch_size": local_params["train_batch_size"],
        "model_name": model_name,
        "max_epoch": local_params["max_epoch"],
        "use_early_stop": local_params["use_early_stop"],
        "num_epoch_early_stop": local_params["num_epoch_early_stop"],
        "main_metric": local_params["main_metric"],
        "use_multi_metrics": local_params["use_multi_metrics"],
        "mutil_metrics": eval(local_params["multi_metrics"]),
        "save_model": local_params["save_model"],
        "accumulation_step": local_params["accumulation_step"]
    }


def config_step_trainer(local_params, global_params, model_name):
    global_params["trainer_config"] = {
        "batch_size": local_params["train_batch_size"],
        "model_name": model_name,
        "max_step": local_params["max_step"],
        "use_early_stop": local_params["use_early_stop"],
        "num_early_stop": local_params["num_early_stop"],
        "num_step2evaluate": local_params["num_step2evaluate"],
        "main_metric": local_params["main_metric"],
        "use_multi_metrics": local_params["use_multi_metrics"],
        "multi_metrics": eval(local_params["multi_metrics"]),
        "save_model": local_params["save_model"],
        "accumulation_step": local_params["accumulation_step"]
    }


def config_exercise_recommendation_trainer(local_params, global_params, model_name):
    config_step_trainer(local_params, global_params, model_name)
    global_params["trainer_config"]["top_ns"] = eval(local_params["top_ns"])


def config_optimizer(local_params, global_params, model_name):
    optimizer_type = local_params[f"optimizer_type"]
    weight_decay = local_params[f"weight_decay"]
    momentum = local_params[f"momentum"]
    learning_rate = local_params[f"learning_rate"]
    enable_scheduler = local_params[f"enable_scheduler"]
    scheduler_type = local_params[f"scheduler_type"]
    scheduler_step = local_params[f"scheduler_step"]
    scheduler_milestones = eval(local_params[f"scheduler_milestones"])
    scheduler_gamma = local_params[f"scheduler_gamma"]
    scheduler_T_max = local_params[f"scheduler_T_max"]
    scheduler_eta_min = local_params[f"scheduler_eta_min"]
    enable_clip_grad = local_params[f"enable_clip_grad"]
    grad_clipped = local_params[f"grad_clipped"]

    global_params["optimizers_config"] = {
        model_name: {}
    }
    optimizer_config = global_params["optimizers_config"][model_name]
    optimizer_config["type"] = optimizer_type
    optimizer_config[optimizer_type] = {}
    optimizer_config[optimizer_type]["lr"] = learning_rate
    optimizer_config[optimizer_type]["weight_decay"] = weight_decay
    if optimizer_type == "sgd":
        optimizer_config[optimizer_type]["momentum"] = momentum

    global_params["schedulers_config"] = {
        model_name: {}
    }
    scheduler_config = global_params["schedulers_config"][model_name]
    if enable_scheduler:
        scheduler_config["use_scheduler"] = True
        scheduler_config["type"] = scheduler_type
        scheduler_config[scheduler_type] = {}
        if scheduler_type == "StepLR":
            scheduler_config[scheduler_type]["step_size"] = scheduler_step
            scheduler_config[scheduler_type]["gamma"] = scheduler_gamma
        elif scheduler_type == "MultiStepLR":
            scheduler_config[scheduler_type]["milestones"] = scheduler_milestones
            scheduler_config[scheduler_type]["gamma"] = scheduler_gamma
        elif scheduler_type == "CosineAnnealingLR":
            scheduler_config[scheduler_type]["T_max"] = scheduler_T_max
            scheduler_config[scheduler_type]["eta_min"] = scheduler_eta_min
        else:
            raise NotImplementedError(f"scheduler `{scheduler_type}` is not implemented")
    else:
        scheduler_config["use_scheduler"] = False

    global_params["grad_clips_config"] = {
        model_name: {}
    }
    grad_clip_config = global_params["grad_clips_config"][model_name]
    grad_clip_config["use_clip"] = enable_clip_grad
    if enable_clip_grad:
        grad_clip_config["grad_clipped"] = grad_clipped

def config_wandb(local_params, global_params, model_name):
    use_wandb = local_params.get("use_wandb", False)
    global_params["trainer_config"]["use_wandb"] = use_wandb
    if use_wandb:
        setting_name = local_params["setting_name"]
        dataset_name = local_params["dataset_name"]
        train_file_name = local_params["train_file_name"]
        wandb.init(project=f"{setting_name}@@{model_name}@@{dataset_name}",
                   name=f"{train_file_name.replace('.txt', '')}@@seed_{local_params['seed']}@@{get_now_time().replace(' ', '@').replace(':', '-')}")
