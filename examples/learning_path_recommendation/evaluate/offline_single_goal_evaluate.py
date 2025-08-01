import argparse
import os

from edmine.utils.data_io import read_kt_file
from edmine.utils.parse import str2bool
from edmine.evaluator.LPREvaluator import LPREvaluator

from config import config_lpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 初始化学生状态的数据集，有些KT模型需要用到一些副信息，所以添加kt_setting_name
    parser.add_argument("--kt_setting_name", type=str, default="pykt_setting")
    parser.add_argument("--setting_name", type=str, default="LPR_offline_setting")
    parser.add_argument("--test_file_name", type=str, default="assist2009_single_goal_test.txt")
    # KT模型配置
    parser.add_argument("--model_dir_name", type=str,
                        default=r"qDKT@@pykt_setting@@assist2009_train@@seed_0@@2025-07-18@20-22-14")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    parser.add_argument("--dataset_name", type=str, default="assist2009", help="for Q table")    
    # kt模型的batch大小
    parser.add_argument("--batch_size", type=int, default=64)
    # 智能体配置
    parser.add_argument("--agent_dir_name", type=str, 
                        help="随机推荐无需训练，仅用agent name表示参数：RandomRecQC-20，随机推荐20个知识点下的习题"
                        "AStarRecConcept-4-5，表示使用A*算法搜索最短学习路径，最多学习4个知识点，每个知识点最多推荐5道习题",
                        default=r"RandomRecQC-20")
    parser.add_argument("--agent_file_name", type=str, help="文件名", default="saved.ckt")
    # 掌握阈值和是否打印学习过程
    parser.add_argument("--master_threshold", type=float, default=0.6)
    parser.add_argument("--render", type=str2bool, default=False)
    # 随机种子
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_log", type=str2bool, default=False)
    parser.add_argument("--save_all_sample", type=str2bool, default=False)
    args = parser.parse_args()
    params = vars(args)
    
    global_params, global_objects = config_lpr(params)
    
    setting_name = params["setting_name"]
    test_file_name = params["test_file_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    test_data = read_kt_file(os.path.join(setting_dir, test_file_name))
    global_objects["data"] = {
        "test": test_data
    }
    
    evaluator = LPREvaluator(global_params, global_objects)
    evaluator.evaluate()
    