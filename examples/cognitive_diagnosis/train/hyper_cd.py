import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.hyper_cd import config_hyper_cd

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.utils.log import get_now_time
from edmine.dataset.CognitiveDiagnosisDataset import BasicCognitiveDiagnosisDataset
from edmine.model.cognitive_diagnosis_model.HyperCD import HyperCD
from edmine.trainer.DLCognitiveDiagnosisTrainer import DLCognitiveDiagnosisTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--evaluate_batch_size", type=int, default=1024)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 模型参数
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--dim_feature", type=int, default=512)
    parser.add_argument("--dim_emb", type=int, default=16)
    parser.add_argument("--leaky", type=float, default=0.8)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_hyper_cd(params)

    global_objects["logger"].info(f"{get_now_time()} start loading and processing dataset")
    dataset_train = BasicCognitiveDiagnosisDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = BasicCognitiveDiagnosisDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "HyperCD": HyperCD(global_params, global_objects).to(global_params["device"])
    }
    trainer = DLCognitiveDiagnosisTrainer(global_params, global_objects)
    trainer.train()
