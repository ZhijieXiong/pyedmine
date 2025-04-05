import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.qikt import config_qikt

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.utils.log import get_now_time
from edmine.dataset.SequentialKTDataset import BasicSequentialKTDataset
from edmine.model.sequential_kt_model.QIKT import QIKT
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


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
    parser.add_argument("--w_q_next_loss", type=float, default=1,
                        help="当且仅当use_irt为False时生效")
    parser.add_argument("--w_c_all_loss", type=float, default=2)
    parser.add_argument("--w_c_next_loss", type=float, default=2)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_qikt(params)

    global_objects["logger"].info(f"{get_now_time()} start loading and processing dataset")
    dataset_train = BasicSequentialKTDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = BasicSequentialKTDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "QIKT": QIKT(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()
