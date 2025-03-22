import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.qdckt import config_qdckt
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.sequential_kt_model.QDCKT import QDCKT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--evaluate_batch_size", type=int, default=1024)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # scheduler配置
    parser.add_argument("--enable_scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler_type", type=str, default="CosineAnnealingLR",
                        choices=("StepLR", "MultiStepLR", "CosineAnnealingLR"))
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_milestones", type=str, default="[5, 10]")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--scheduler_T_max", type=int, default=10)
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0001)
    # 模型参数
    parser.add_argument("--dim_emb", type=int, default=64)
    parser.add_argument("--dim_correctness", type=int, default=64)
    parser.add_argument("--dim_latent", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=11)
    parser.add_argument("--rnn_type", type=str, default="gru")
    parser.add_argument("--num_rnn_layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_predict_layer", type=int, default=2)
    parser.add_argument("--dim_predict_mid", type=int, default=512)
    parser.add_argument("--activate_type", type=str, default="sigmoid")
    parser.add_argument("--w_qdckt_loss", type=float, default=0.1)

    # 设置参数空间
    parameters_space = {
        "window_size": [1, 11, 21],
        "w_qdckt_loss": [0.01, 0.1, 1],
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
    fmin(get_objective_func(parser, config_qdckt, "QDCKT", QDCKT), space, algo=tpe.suggest, max_evals=max_evals)
