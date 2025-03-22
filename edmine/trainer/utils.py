import torch.optim as optim


def create_optimizer(parameters, opt_config):
    if opt_config["type"] == 'sgd':
        optimizer = optim.SGD(parameters, **opt_config["sgd"])
    elif opt_config["type"] == 'adamW':
        optimizer = optim.AdamW(parameters, **opt_config["adamW"])
    else:
        optimizer = optim.Adam(parameters, **opt_config["adam"])
    return optimizer


def create_scheduler(optimizer, sch_config):
    if sch_config["type"] == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sch_config["CosineAnnealingLR"])
    elif sch_config["type"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sch_config["MultiStepLR"])
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sch_config["StepLR"])
    return scheduler
