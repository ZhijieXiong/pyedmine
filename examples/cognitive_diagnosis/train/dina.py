import argparse

from torch.utils.data import DataLoader

from set_params import *
from config.dina import config_dina

from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.dataset.CognitiveDiagnosisDataset import BasicCognitiveDiagnosisDataset
from edmine.model.cognitive_diagnosis_model.DINA import DINA
from edmine.trainer.DLCognitiveDiagnosisTrainer import DLCognitiveDiagnosisTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args(), setup_scheduler_args(), setup_clip_args(), setup_grad_acc_args()], 
                                     add_help=False)
    # batch size
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--evaluate_batch_size", type=int, default=512)
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adam", choices=("adam", "sgd"))
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    # 模型参数
    parser.add_argument("--max_slip", type=float, default=0.4)
    parser.add_argument("--max_guess", type=float, default=0.4)
    parser.add_argument("--max_step", type=int, default=1000)
    parser.add_argument("--use_ste", type=str2bool, default=True)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])
    global_params, global_objects = config_dina(params)

    dataset_train = BasicCognitiveDiagnosisDataset(global_params["datasets_config"]["train"], global_objects)
    dataloader_train = DataLoader(dataset_train, batch_size=params["train_batch_size"], shuffle=True)
    dataset_valid = BasicCognitiveDiagnosisDataset(global_params["datasets_config"]["valid"], global_objects)
    dataloader_valid = DataLoader(dataset_valid, batch_size=params["train_batch_size"], shuffle=False)

    global_objects["data_loaders"] = {
        "train_loader": dataloader_train,
        "valid_loader": dataloader_valid
    }
    global_objects["models"] = {
        "DINA": DINA(global_params, global_objects).to(global_params["device"])
    }
    trainer = DLCognitiveDiagnosisTrainer(global_params, global_objects)
    trainer.train()
