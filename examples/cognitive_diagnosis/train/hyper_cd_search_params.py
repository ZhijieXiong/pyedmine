import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.hyper_cd import config_hyper_cd
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.cognitive_diagnosis_model.HyperCD import HyperCD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--evaluate_batch_size", type=int, default=1024)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 模型参数
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--dim_feature", type=int, default=512)
    parser.add_argument("--dim_emb", type=int, default=16)
    parser.add_argument("--leaky", type=float, default=0.8)

    # 设置参数空间
    parameters_space = {
        "num_layer": [3, 4],
        "dim_feature": [512, 1024],
        "dim_emb": [8, 16],
    }
    space = {
        param_name: hp.choice(param_name, param_space)
        for param_name, param_space in parameters_space.items()
    }
    num = 1
    for parameter_space in parameters_space.values():
        num *= len(parameter_space)
    if num > 100:
        max_evals = 20 + int(num * 0.2)
    elif num > 50:
        max_evals = 15 + int(num * 0.2)
    elif num > 20:
        max_evals = 10 + int(num * 0.2)
    elif num > 10:
        max_evals = 5 + int(num * 0.2)
    else:
        max_evals = num
    current_best_performance = 0
    fmin(get_objective_func(parser, config_hyper_cd, "HyperCD", HyperCD), space, algo=tpe.suggest, max_evals=max_evals)

