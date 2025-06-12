import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.hawkes_kt import config_hawkes_kt
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.sequential_kt_model.HawkesKT import HawkesKT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 模型参数
    parser.add_argument("--dim_emb", type=int, default=64)
    parser.add_argument("--time_log", type=int, default=5)
    # 是否自动裁剪batch序列
    parser.add_argument("--auto_clip_seq", type=str2bool, default=False)

    # 设置参数空间
    parameters_space = {
        "weight_decay": [0.0001, 0.00001, 0.000001, 0],
        "dim_emb": [64, 128, 256],
        "time_log": [2, 5, 10]
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
    fmin(get_objective_func(parser, config_hawkes_kt, "HawkesKT", HawkesKT), space, algo=tpe.suggest, max_evals=max_evals)
