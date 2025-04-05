import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.clkt import config_clkt

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.utils.log import get_now_time
from edmine.dataset.SequentialKTDatasetWithSample import CLKTDataset
from edmine.model.sequential_kt_model.CLKT import CLKT
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_grad_acc_args(), setup_clip_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=128)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 数据增强
    parser.add_argument("--mask_prob", type=float, default=0.3)
    parser.add_argument("--replace_prob", type=float, default=0.3)
    parser.add_argument("--crop_prob", type=float, default=0.3)
    parser.add_argument("--permute_prob", type=float, default=0.3)
    # 对比学习
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--w_cl_loss", type=float, default=0.1)
    # 模型参数
    parser.add_argument("--dim_model", type=int, default=64)
    parser.add_argument("--num_block", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dim_final_fc", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--key_query_same", type=str2bool, default=True)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_clkt(params)

    global_objects["logger"].info(f"{get_now_time()} start loading and processing dataset")
    dataset_train = CLKTDataset(global_params["datasets_config"]["train"], global_objects, train_mode=True)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = CLKTDataset(global_params["datasets_config"]["valid"], global_objects, train_mode=False)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "CLKT": CLKT(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()
