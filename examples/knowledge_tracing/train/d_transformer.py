import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.d_transformer import config_d_transformer

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.dataset.SequentialKTDataset import DTransformerDataset
from edmine.model.sequential_kt_model.DTransformer import DTransformer
from edmine.trainer.SequentialDLKTTrainer import SequentialDLKTTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--evaluate_batch_size", type=int, default=32)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=True)
    parser.add_argument("--grad_clipped", type=float, default=1.0)
    # 梯度累计
    parser.add_argument("--accumulation_step", type=int, default=2,
                        help="1表示不使用，大于1表示使用accumulation_step的梯度累计")
    # 模型参数
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--num_know", type=int, default=32)
    parser.add_argument("--dim_final_fc", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--key_query_same", type=str2bool, default=True)
    # 对比损失
    parser.add_argument("--temperature", type=float, default=0.05, help="官方代码直接固定为0.05")
    parser.add_argument("--w_cl_loss", type=float, default=0.1)
    parser.add_argument("--w_reg_loss", type=float, default=0.001, help="官方代码直接固定为0.001")
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_d_transformer(params)

    dataset_train = DTransformerDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = DTransformerDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "DTransformer": DTransformer(global_params, global_objects).to(global_params["device"])
    }
    trainer = SequentialDLKTTrainer(global_params, global_objects)
    trainer.train()
