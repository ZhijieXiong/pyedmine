import argparse
from hyperopt import fmin, tpe, hp

from set_params.congnitive_diagnosis_params import setup_common_args
from config.ncd import config_ncd
from utils import get_objective_func

from edmine.utils.parse import str2bool
from edmine.model.cognitive_diagnosis_model.NCD import NCD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args()], add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--evaluate_batch_size", type=int, default=2048)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # scheduler配置
    parser.add_argument("--enable_scheduler", type=str2bool, default=False)
    parser.add_argument("--scheduler_type", type=str, default="MultiStepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_milestones", type=str, default="[5, 10]")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 梯度累计
    parser.add_argument("--accumulation_step", type=int, default=1,
                        help="1表示不使用，大于1表示使用accumulation_step的梯度累计")
    # 模型参数
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_predict_layer", type=int, default=2)
    parser.add_argument("--dim_predict_mid", type=int, default=64)
    parser.add_argument("--activate_type", type=str, default="sigmoid")

    # 设置参数空间
    parameters_space = {
        # 一阶段
        # "train_batch_size": [512, 1024, 2048],
        # "learning_rate": [0.0001, 0.001],
        # "weight_decay": [0.0001, 0.00001, 0],
        # 二阶段
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
    fmin(get_objective_func(parser, config_ncd, "NCD", NCD), space, algo=tpe.suggest, max_evals=max_evals)

