import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.akt import config_akt
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.sequential_kt_model.AKT import AKT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=True)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 模型参数
    parser.add_argument("--dim_model", type=int, default=256)
    parser.add_argument("--num_block", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_predict_layer", type=int, default=2)
    parser.add_argument("--dim_predict_mid", type=int, default=64)
    parser.add_argument("--activate_type", type=str, default="relu")
    parser.add_argument("--key_query_same", type=str2bool, default=True)
    parser.add_argument("--separate_qa", type=str2bool, default=False)
    parser.add_argument("--w_rasch_loss", type=float, default=0.00001)
    # 是否自动裁剪batch序列
    parser.add_argument("--auto_clip_seq", type=str2bool, default=False)

    # 设置参数空间
    parameters_space = {
        "dim_model": [64, 256],
        "num_block": [1, 2, 4],
        "num_head": [4, 8],
        "dim_ff": [64, 256],
        "dropout": [0.1, 0.2, 0.3]
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
    fmin(get_objective_func(parser, config_akt, "AKT", AKT), space, algo=tpe.suggest, max_evals=max_evals)
