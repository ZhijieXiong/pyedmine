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
                        default="RCD@@ncd_setting@@assist2009_train_fold_0@@seed_0@@2025-03-07@03-24-40")
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test.txt")
    parser.add_argument("--evaluate_batch_size", type=int, default=4096)
    # 冷启动问题
    parser.add_argument("--user_cold_start", type=int, default=10,
                        help="大于0则开启user冷启动评估")
    parser.add_argument("--question_cold_start", type=int, default=10,
                        help="大于0则开启question冷启动评估")
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
