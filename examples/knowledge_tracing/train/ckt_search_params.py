import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.ckt import config_ckt
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.sequential_kt_model.CKT import CKT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 学习率衰减
    parser.add_argument("--enable_scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_milestones", type=str, default="[10, 20]")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--scheduler_T_max", type=int, default=10)
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0001)
    # 模型参数
    parser.add_argument("--dim_emb", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kernel_size", type=int, default=7)

    # 设置参数空间
    parameters_space = {
        "num_layer": [1, 2, 4, 6],
        "kernel_size": [3, 5, 7, 9],
        "dropout": [0.1, 0.2, 0.3],
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
    fmin(get_objective_func(parser, config_ckt, "CKT", CKT), space, algo=tpe.suggest, max_evals=max_evals)
