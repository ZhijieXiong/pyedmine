import argparse
import os

from set_params import *
from config.a2c import config_a2c

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.utils.log import get_now_time
from edmine.utils.data_io import read_kt_file
from edmine.trainer.LPROnlineDRLTrainer import LPROnlineDRLTrainer
from edmine.model.learning_path_recommendation_agent.A2C import A2C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_step_trainer_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 折扣因子
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    # 模型参数
    parser.add_argument("--max_question_attempt", type=int, default=20)
    parser.add_argument("--num_layer_action_model", type=int, default=2)
    parser.add_argument("--num_layer_state_model", type=int, default=2)
    parser.add_argument("--interval_step", type=int, default=100)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_a2c(params)

    global_objects["logger"].info(f"{get_now_time()} start loading and processing dataset")
    setting_dir = global_objects["file_manager"].get_setting_dir(params["setting_name"])
    global_objects["data"] = {
        "train": read_kt_file(os.path.join(setting_dir, params["train_file_name"])),
        "valid": read_kt_file(os.path.join(setting_dir, params["valid_file_name"]))
    }

    agent = A2C(global_params, global_objects)
    global_objects["agents"] = {
        "A2C": agent
    }
    
    # 每隔interval_step步都执行一次copy_state_model
    global_params["trainer_config"]["interval_execute_config"] = {
        params["interval_step"]: [agent.copy_state_model]
    }
    
    trainer = LPROnlineDRLTrainer(global_params, global_objects)
    trainer.train()
