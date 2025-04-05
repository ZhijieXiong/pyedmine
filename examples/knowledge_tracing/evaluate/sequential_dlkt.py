import argparse
from torch.utils.data import DataLoader

from config import config_sequential_dlkt
from utils import get_model_info, select_dataset

from edmine.utils.parse import str2bool
from edmine.evaluator.SequentialDLKTEvaluator import SequentialDLKTEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 加载模型参数配置（如果不修改trainer中save代码，那么只需要修改model_dir_name参数）
    parser.add_argument("--model_dir_name", type=str, help="",
                        default="DKT@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-06@02-12-29")
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置（通常只需要修改dataset_name和test_file_name参数）
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test.txt")
    parser.add_argument("--seq_start", type=int, default=2, help="序列中seq_start（自然序列，从1开始）之前的元素不参与评估")
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # ===============================不同场景指标设置===============================
    # 常规测试
    parser.add_argument("--evaluate_overall", type=str2bool, default=True)
    # 冷启动问题
    parser.add_argument("--question_cold_start", type=int, default=-1,
                        help="大于等于0则开启冷启动评估，即评估在训练数据集中出现次数少于等于k个的习题的预测结果")
    parser.add_argument("--user_cold_start", type=int, default=0,
                        help="大于等于1则开启冷启动评估，即评估每个用户前k个预测结果")
    # 多步测试（参照PYKT-paper 4.1 Observation 5实现）
    parser.add_argument("--multi_step", type=int, default=1,
                        help="大于等于2则开启多步测试")
    parser.add_argument("--multi_step_accumulate", type=str2bool, default=False)
    # core指标
    parser.add_argument("--use_core", type=str2bool, default=False)
    # ===========================================================================
    # 是否保存每个样本的测试结果
    parser.add_argument("--save_all_sample", type=str2bool, default=False)
    # 是否保存测试结果
    parser.add_argument("--save_log", type=str2bool, default=False)
    # 随机种子（有些指标带有随机性）
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_sequential_dlkt(params)
    model_name, setting_name, _ = get_model_info(params["model_dir_name"])

    dataset_test = select_dataset(model_name)({
        "setting_name": setting_name,
        "file_name": params["test_file_name"],
        "device": global_params["device"]
    }, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)
    global_objects["data_loaders"] = {"test_loader": dataloader_test}
    evaluator = SequentialDLKTEvaluator(global_params, global_objects)
    evaluator.evaluate()
