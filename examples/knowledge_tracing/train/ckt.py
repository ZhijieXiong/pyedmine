import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.ckt import config_ckt

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.dataset.SequentialKTDataset import CKTDataset
from edmine.model.sequential_kt_model.CKT import CKT
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--evaluate_batch_size", type=int, default=64)
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
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_ckt(params)

    dataset_train = CKTDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = CKTDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "CKT": CKT(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()
