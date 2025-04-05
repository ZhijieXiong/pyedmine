import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from edmine.utils.parse import str2bool
from config.abqr import config_abqr
from utils import get_objective_func

from edmine.model.sequential_kt_model.ABQR import ABQR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=80)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=True)
    parser.add_argument("--grad_clipped", type=float, default=15.0)
    # 模型参数
    parser.add_argument("--dim_emb", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # 设置参数空间
    parameters_space = {
        "dim_emb": [64, 128, 256],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
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
    fmin(get_objective_func(parser, config_abqr, "ABQR", ABQR), space, algo=tpe.suggest, max_evals=max_evals)
