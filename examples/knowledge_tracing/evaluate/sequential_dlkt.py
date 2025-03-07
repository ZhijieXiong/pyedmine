import argparse
from torch.utils.data import DataLoader

from config import config_sequential_dlkt

from edmine.utils.parse import str2bool
from edmine.dataset.SequentialKTDataset import BasicSequentialKTDataset
from edmine.evaluator.SequentialDLKTEvaluator import SequentialDLKTEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 加载模型参数配置
    parser.add_argument("--model_dir_name", type=str, help="",
                        default="DKVMN@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-06@02-12-53")
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test.txt")
    parser.add_argument("--seq_start", type=int, default=2, help="序列中seq_start之前的元素不参与评估")
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 冷启动问题
    parser.add_argument("--cold_start", type=int, default=10,
                        help="大于等于1则开启冷启动评估")
    # 多步测试（参照PYKT-paper 4.1 Observation 5实现）
    parser.add_argument("--multi_step", type=int, default=10,
                        help="大于等于2则开启多步测试")
    # 保存测试结果
    parser.add_argument("--save_log", type=str2bool, default=True)
    # 随机种子（有些指标带有随机性）
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_sequential_dlkt(params)

    dataset_test = BasicSequentialKTDataset({
        "setting_name": params["setting_name"],
        "file_name": params["test_file_name"],
        "device": global_params["device"]
    }, global_objects)
    dataloader_test = DataLoader(dataset_test, batch_size=params["evaluate_batch_size"], shuffle=False)
    global_objects["data_loaders"] = {"test_loader": dataloader_test}
    evaluator = SequentialDLKTEvaluator(global_params, global_objects)
    evaluator.evaluate()
