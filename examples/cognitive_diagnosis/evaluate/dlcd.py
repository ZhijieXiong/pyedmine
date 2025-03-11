import argparse
from torch.utils.data import DataLoader

from config import config_dlcd
from utils import get_model_info

from edmine.utils.parse import str2bool
from edmine.dataset.CognitiveDiagnosisDataset import BasicCognitiveDiagnosisDataset
from edmine.evaluator.DLCDEvaluator import DLCDEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 加载模型参数配置
    parser.add_argument("--model_dir_name", type=str, help="",
                        default="NCD@@ncd_setting@@assist2009_train_fold_0@@seed_0@@2025-03-11@01-46-58")
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test.txt")
    parser.add_argument("--evaluate_batch_size", type=int, default=2048)
    # 冷启动问题
    parser.add_argument("--evaluate_overall", type=str2bool, default=True)
    parser.add_argument("--user_cold_start", type=int, default=-1,
                        help="大于等于0则开启冷启动评估，即评估在训练数据集中练习记录数量小于等于k个的用户预测结果")
    parser.add_argument("--question_cold_start", type=int, default=-1,
                        help="大于等于0则开启冷启动评估，即评估在训练数据集中出现次数小于等于k个的习题的预测结果")
    # 保存测试结果
    parser.add_argument("--save_log", type=str2bool, default=True)

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_dlcd(params)

    dataset_test = BasicCognitiveDiagnosisDataset({
        "setting_name": get_model_info(params["model_dir_name"])[1],
        "file_name": params["test_file_name"],
        "device": global_params["device"]
    }, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)
    global_objects["data_loaders"] = {"test_loader": dataloader_test}
    evaluator = DLCDEvaluator(global_params, global_objects)
    evaluator.evaluate()
