import argparse

from edmine.utils.data_io import read_kt_file
from edmine.utils.parse import str2bool
from edmine.evaluator.LPREvaluator import LPREvaluator

from config import config_lpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 初始化学生状态的数据集
    parser.add_argument("--kt_setting_name", type=str, default="pykt_setting")
    parser.add_argument("--setting_name", type=str, default="LPR_offline_setting")
    parser.add_argument("--test_file_name", type=str, default="assist2009_single_goal_test.txt")
    # 模拟器配置
    parser.add_argument("--model_dir_name", type=str,
                        default=r"MIKT4LPR@@pykt_setting@@assist2009_train@@seed_0@@2025-07-18@15-04-22")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    parser.add_argument("--dataset_name", type=str, default="assist2009", help="for Q table")
    # 掌握阈值
    parser.add_argument("--master_threshold", type=float, default=0.6)
    # kt模型的batch大小
    parser.add_argument("--batch_size", type=int, default=64)
    # 智能体配置
    parser.add_argument("--agent_dir_name", type=str, 
                        help="RandomAgent@@random-5@@10，random-5是rec concept的策略，10表示每个知识点最多推荐10道习题"
                        "RandomAgent@@AStar-5@@10，AStar-5是rec concept的策略，表示使用A*算法搜索最短学习路径，最多学习5个知识点，10表示每个知识点最多推荐10道习题",
                        default=r"RandomAgent@@AStar-5@@10")
    # 随机种子
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_log", type=str2bool, default=False)
    parser.add_argument("--save_all_sample", type=str2bool, default=False)
    args = parser.parse_args()
    params = vars(args)
    
    global_params, global_objects = config_lpr(params)
    
    test_data = read_kt_file(global_params["datasets_config"]["test"]["file_path"])
    global_objects["datasets"] = {
        "test": test_data
    }
    
    evaluator = LPREvaluator(global_params, global_objects)
    evaluator.evaluate()
    