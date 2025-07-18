import argparse
from torch.utils.data import DataLoader

from config import config_sequential_dlkt
from utils import get_model_info, select_dataset

from edmine.utils.parse import str2bool
from edmine.evaluator.SequentialDLKTEvaluator4FTAcc import SequentialDLKTEvaluator4FTAcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 加载模型参数配置
    parser.add_argument("--model_dir_name", type=str, help="",
                        default="DKT@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-06@02-12-29")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--test_file_name", type=str, help="文件名", default="assist2009_test.txt")
    parser.add_argument("--seq_start", type=int, default=2, help="序列中seq_start（自然序列，从1开始）之前的元素不参与评估")
    parser.add_argument("--evaluate_batch_size", type=int, default=256)
    # 保存测试结果
    parser.add_argument("--save_log", type=str2bool, default=False)

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
    evaluator = SequentialDLKTEvaluator4FTAcc(global_params, global_objects)
    evaluator.evaluate()
