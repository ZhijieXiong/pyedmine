import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.qikt import config_qikt
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.sequential_kt_model.QIKT import QIKT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--evaluate_batch_size", type=int, default=128)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 模型参数
    parser.add_argument("--dim_emb", type=int, default=256)
    parser.add_argument("--rnn_type", type=str, default="gru",
                        choices=("rnn", "lstm", "gru"))
    parser.add_argument("--num_rnn_layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_mlp_layer", type=int, default=1)
    parser.add_argument("--lambda_q_all", type=int, default=1)
    parser.add_argument("--lambda_c_next", type=int, default=1)
    parser.add_argument("--lambda_c_all", type=int, default=1)
    parser.add_argument("--use_irt", type=str2bool, default=True)
    # 损失权重
    parser.add_argument("--w_q_all_loss", type=float, default=1.5)
    parser.add_argument("--w_q_next_loss", type=float, default=1)
    parser.add_argument("--w_c_all_loss", type=float, default=2)
    parser.add_argument("--w_c_next_loss", type=float, default=2)

    # 设置参数空间
    parameters_space = {
        "weight_decay": [0.00001, 0],
        "dim_emb": [64, 256],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "w_q_all_loss": [0.5, 1, 2],
        "w_c_all_loss": [0.5, 1, 2],
        "w_c_next_loss": [0.5, 1, 2],
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
    fmin(get_objective_func(parser, config_qikt, "QIKT", QIKT), space, algo=tpe.suggest, max_evals=max_evals)
