import argparse
from hyperopt import fmin, tpe, hp

from set_params import *
from config.d3qn import config_d3qn
from utils import get_objective_func

from edmine.model.learning_path_recommendation_agent.D3QNAgent import D3QNAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--train_batch_size", type=int, default=64)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 折扣因子和探索概率
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="ε-greedy")
    # 模型参数
    parser.add_argument("--max_question_attempt", type=int, default=20)
    parser.add_argument("--dim_c_feature", type=int, default=64)
    parser.add_argument("--dim_q_feature", type=int, default=128)
    parser.add_argument("--num_layer_c_rec_model", type=int, default=2)
    parser.add_argument("--num_layer_q_rec_model", type=int, default=2)
    
    # 设置参数空间
    parameters_space = {
        "buffer_size": [500, 1000, 2000],
        "learning_rate": [0.00001, 0.0001, 0.001],
        "train_batch_size": [16, 32, 64]
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
    fmin(get_objective_func(parser, config_d3qn, "D3QN", D3QNAgent), space, algo=tpe.suggest, max_evals=max_evals)
