import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.hdlpkt import config_hdlpkt
from utils import auto_clip_seq

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.utils.log import get_now_time
from edmine.dataset.SequentialKTDataset import HDLPKTDataset
from edmine.model.sequential_kt_model.HDLPKT import HDLPKT
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


# HDLPKT跑不了assist2009，在setup_common_args中调整数据集
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--evaluate_batch_size", type=int, default=8)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 梯度累计
    parser.add_argument("--accumulation_step", type=int, default=8,
                        help="1表示不使用，大于1表示使用accumulation_step的梯度累计")
    # 模型参数
    parser.add_argument("--dim_k", type=int, default=128)
    parser.add_argument("--dim_e", type=int, default=128)
    parser.add_argument("--dim_a", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_seq_length", type=int, default=200)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    # 是否自动裁剪batch序列
    parser.add_argument("--auto_clip_seq", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_hdlpkt(params)

    global_objects["logger"].info(f"{get_now_time()} start loading and processing dataset")
    if params["auto_clip_seq"]:
        collate_fn = auto_clip_seq
    else:
        collate_fn = None
    dataset_train = HDLPKTDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True, collate_fn=collate_fn)
    dataset_valid = HDLPKTDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False, collate_fn=collate_fn)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "HDLPKT": HDLPKT(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()
