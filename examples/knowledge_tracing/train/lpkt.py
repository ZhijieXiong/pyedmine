import argparse

from torch.utils.data import DataLoader

from set_params.sequential_kt_params import setup_common_args
from config.lpkt import config_lpkt

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.dataset.SequentialKTDataset import LPKTDataset
from edmine.model.sequential_kt_model.LPKT import LPKT
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args()], add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # scheduler配置
    parser.add_argument("--enable_scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler_type", type=str, default="StepLR",
                        choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--scheduler_step", type=int, default=15)
    parser.add_argument("--scheduler_milestones", type=str, default="[5, 10]")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 梯度累计
    parser.add_argument("--accumulation_step", type=int, default=1,
                        help="1表示不使用，大于1表示使用accumulation_step的梯度累计")
    # 模型参数
    parser.add_argument("--dim_k", type=int, default=64)
    parser.add_argument("--dim_e", type=int, default=64)
    parser.add_argument("--dim_correctness", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_lpkt(params)

    dataset_train = LPKTDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = LPKTDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "LPKT": LPKT(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()
