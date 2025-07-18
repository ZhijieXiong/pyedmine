import argparse

from utils import get_model_info
from config import config_dler

from edmine.utils.data_io import read_kt_file, read_mlkc_data
from edmine.utils.parse import str2bool
from edmine.dataset.KG4EXDataset import *
from edmine.evaluator.DLEREvaluator import DLEREvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 加载模型参数配置
    parser.add_argument("--model_dir_name", type=str, help="",
                        default="KG4EX@@ER_offline_setting@@assist2009_train_triples_dkt_0.2@@seed_0@@2025-03-13@20-19-52")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 测试配置
    parser.add_argument("--top_ns", type=str, default="[5,10,20]")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--user_data_file_name", type=str, default="assist2009_user_data.txt")
    parser.add_argument("--test_mlkc_file_name", type=str, default="assist2009_dkt_mlkc_test.txt")
    parser.add_argument("--test_pkc_file_name", type=str, default="assist2009_pkc_test.txt")
    parser.add_argument("--test_efr_file_name", type=str, default="assist2009_efr_0.2_test.txt")
    parser.add_argument("--evaluate_batch_size", type=int, default=4)    
    # 保存测试结果
    parser.add_argument("--save_log", type=str2bool, default=True)
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_dler(params)

    setting_name = get_model_info(params["model_dir_name"])[1]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    kg4ex_dir = os.path.join(setting_dir, "KG4EX")
    users_data = read_kt_file(os.path.join(setting_dir, params['user_data_file_name']))
    users_data_dict = {}
    for user_data in users_data:
        users_data_dict[user_data["user_id"]] = user_data
    global_objects["data_loaders"] = {
        # users_data_dict和mlkc是计算指标时需要的数据，所有推荐模型都要，第3个元素则是各个模型推理时需要的数据
        "test_loader": (users_data_dict,
                         read_mlkc_data(os.path.join(kg4ex_dir, params["test_mlkc_file_name"])),
                         (read_mlkc_data(os.path.join(kg4ex_dir, params["test_pkc_file_name"])),
                          read_mlkc_data(os.path.join(kg4ex_dir, params["test_efr_file_name"]))))
    }
    evaluator = DLEREvaluator(global_params, global_objects)
    evaluator.evaluate()

