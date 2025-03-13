import argparse

from torch.utils.data import DataLoader

from config.kg4ex import config_kg4ex
from set_params.exercise_recommendation_params import setup_common_args

from edmine.utils.data_io import read_kt_file, read_mlkc_data
from edmine.utils.parse import str2bool
from edmine.utils.use_torch import set_seed
from edmine.dataset.KG4EXDataset import *
from edmine.model.exercise_recommendation_model.KG4EX import KG4EX
from edmine.trainer.ExerciseRecommendationTrainer import ExerciseRecommendationTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[setup_common_args()], add_help=False)
    # 数据集
    parser.add_argument("--valid_pkc_file_name", type=str, default="assist2009_pkc_valid.txt")
    parser.add_argument("--valid_efr_file_name", type=str, default="assist2009_efr_0.2_valid.txt")
    # 学习率
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--evaluate_batch_size", type=int, default=1, 
                        help="推理比较占显存，与知识点数量相关")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--enable_scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler_type", type=str, default="MultiStepLR", choices=("StepLR", "MultiStepLR"))
    parser.add_argument("--scheduler_step", type=int, default=2000, help="unit: step")
    parser.add_argument("--scheduler_milestones", type=str, default="[1000, 3000, 9000]", help="unit: step")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    # 梯度裁剪
    parser.add_argument("--enable_clip_grad", type=str2bool, default=False)
    parser.add_argument("--grad_clipped", type=float, default=10.0)
    # 梯度累计
    parser.add_argument("--accumulation_step", type=int, default=1,
                        help="1表示不使用，大于1表示使用accumulation_step的梯度累计")
    # 模型参数
    parser.add_argument("--negative_sample_size", type=int, default=256)
    parser.add_argument("--model_selection", type=str, default="TransE", choices=('TransE', 'RotatE'))
    parser.add_argument("--dim", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=12)
    parser.add_argument("--double_entity_embedding", type=str2bool, default=True)
    parser.add_argument("--double_relation_embedding", type=str2bool, default=True)
    parser.add_argument("--negative_adversarial_sampling", type=str2bool, default=True)
    parser.add_argument("--uni_weight", type=str2bool, default=True)
    parser.add_argument("--adversarial_temperature", type=float, default=1)
    parser.add_argument("--epsilon", type=float, default=2)
    parser.add_argument("--w_reg_loss", type=float, default=0)
    # 其它
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--use_wandb", type=str2bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    set_seed(params["seed"])

    global_params, global_objects = config_kg4ex(params)

    setting_name = params["setting_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    kg4ex_dir = os.path.join(setting_dir, "kg4ex")
    dataset_head_train = KG4EXDataset(global_params["datasets_config"]["train"]["head"], global_objects)
    dataset_tail_train = KG4EXDataset(global_params["datasets_config"]["train"]["tail"], global_objects)
    dataloader_head_train = DataLoader(dataset_head_train, batch_size=params["train_batch_size"], shuffle=True)
    dataloader_tail_train = DataLoader(dataset_tail_train, batch_size=params["train_batch_size"], shuffle=True)
    train_iterator = BidirectionalOneShotIterator(dataloader_head_train, dataloader_tail_train)

    users_data = read_kt_file(os.path.join(setting_dir, params['user_data_file_name']))
    users_data_dict = {}
    for user_data in users_data:
        users_data_dict[user_data["user_id"]] = user_data
    global_objects["data_loaders"] = {
        "train_loader": train_iterator,
        # users_data_dict和mlkc是计算指标时需要的数据，所有推荐模型都要，第3个元素则是各个模型推理时需要的数据
        "valid_loader": (users_data_dict,
                         read_mlkc_data(os.path.join(kg4ex_dir, params["valid_mlkc_file_name"])),
                         (read_mlkc_data(os.path.join(kg4ex_dir, params["valid_pkc_file_name"])),
                          read_mlkc_data(os.path.join(kg4ex_dir, params["valid_efr_file_name"]))))
    }
    global_objects["models"] = {"KG4EX": KG4EX(global_params, global_objects).to(global_params["device"])}
    trainer = ExerciseRecommendationTrainer(global_params, global_objects)
    trainer.train()
