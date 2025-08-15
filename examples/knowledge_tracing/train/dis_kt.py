import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.dis_kt import config_dis_kt

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.utils.log import get_now_time
from edmine.dataset.SequentialKTDatasetWithSample import DisKTDataset
from edmine.model.sequential_kt_model.DisKT import DisKT
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--evaluate_batch_size", type=int, default=1024)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=True)
    parser.add_argument("--grad_clipped", type=float, default=2.0)
    # 模型参数
    parser.add_argument("--dim_model", type=int, default=64)
    parser.add_argument("--num_block", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num_predict_layer", type=int, default=2)
    parser.add_argument("--dim_predict_mid", type=int, default=64)
    parser.add_argument("--activate_type", type=str, default="relu")
    parser.add_argument("--key_query_same", type=str2bool, default=True)
    parser.add_argument("--separate_qa", type=str2bool, default=False)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--w_reg_loss", type=float, default=0.001)
    parser.add_argument("--neg_prob", type=float, default=0.2)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_dis_kt(params)

    global_objects["logger"].info(f"{get_now_time()} start loading and processing dataset")
    dataset_train = DisKTDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = DisKTDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "DisKT": DisKT(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()