import argparse
import os
import numpy as np

import config
from utils import get_performance
from rec_strategy import *

from edmine.utils.data_io import read_mlkc_data, read_kt_file
from edmine.utils.parse import q2c_from_q_table, cal_qc_acc4kt_data
from edmine.data.FileManager import FileManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--user_data_file_name", type=str, default="assist2009_user_data.txt")
    parser.add_argument("--user_sim_mat_file_name", type=str, default="assist2009_cossim_1_1_1.npy")
    # 评价指标选择
    parser.add_argument("--used_metrics", type=str, default="['KG4EX_ACC', 'KG4EX_NOV', 'PERSONALIZATION_INDEX', 'OFFLINE_ACC', 'OFFLINE_NDCG']",
                        help='KG4EX_ACC, KG4EX_VOL, PERSONALIZATION_INDEX, OFFLINE_ACC, OFFLINE_NDCG')
    parser.add_argument("--top_ns", type=str, default="[5,10,20]")
    # KG4EX_ACC指标需要的数据
    parser.add_argument("--mlkc_file_name", type=str, default="assist2009_dkt_mlkc_test.txt")
    parser.add_argument("--delta", type=float, default=0.7)
    # 推荐策略
    parser.add_argument("--rec_strategy", type=int, default=0,
                        help="0: 从相似用户的练习习题中找出用户未练习过的习题，并根据用户的难度偏好（如历史平均难度 ± 阈值）筛选习题")
    parser.add_argument("--method0_diff_threshold", type=float, default=0.1,
                        help="method 0过滤习题时的难度阈值")
    # 其它
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    params = vars(args)
    np.random.seed(params["seed"])

    setting_name = params["setting_name"]
    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(setting_name)
    save_dir = os.path.join(setting_dir, "user_smi_mat")
    users_data = read_kt_file(os.path.join(setting_dir, params["user_data_file_name"]))
    Q_table = file_manager.get_q_table(params["dataset_name"])
    q2c = q2c_from_q_table(Q_table)
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]

    question_acc = cal_qc_acc4kt_data(users_data, "question", 0)
    average_que_acc = sum(question_acc.values()) / len(question_acc)
    question_diff = {}
    for q_id in range(num_question):
        if q_id not in question_acc:
            question_diff[q_id] = average_que_acc
        else:
            question_diff[q_id] = 1 - question_acc[q_id]

    user_sim_mat = np.load(os.path.join(save_dir, params["user_sim_mat_file_name"]))
    similar_users = np.argsort(-user_sim_mat, axis=1)

    rec_strategy = params["rec_strategy"]
    top_ns = eval(params["top_ns"])
    rec_result = {x: {} for x in top_ns}
    if rec_strategy == 0:
        top_ns = sorted(top_ns, reverse=True)
        last_top_n = top_ns[0]
        for i, top_n in enumerate(top_ns):
            if i == 0:
                rec_result[top_n] = rec_method_based_on_user_sim(
                    users_data, similar_users, question_diff, params["method0_diff_threshold"], top_n)
            else:
                rec_result[top_n] = {
                    user_id: rec_ques[:top_n] for user_id, rec_ques in rec_result[last_top_n].items()}
                last_top_n = top_n
    else:
        raise ValueError(f"{rec_strategy} is not implemented")

    used_metrics = eval(params["used_metrics"])
    mlkc = read_mlkc_data(os.path.join(setting_dir, params["mlkc_file_name"]))
    performance = get_performance(used_metrics, top_ns, users_data, rec_result, q2c, mlkc, params["delta"])
    top_ns = sorted(top_ns)
    print(f"performance of {params['user_sim_mat_file_name']}-rec{params['rec_strategy']}")
    for top_n in top_ns:
        top_n_performance = performance[top_n]
        performance_str = ""
        for metric_name, metric_value in top_n_performance.items():
            performance_str += f"{metric_name}: {metric_value:<9.5}, "
        print(f"    top {top_n} performances are {performance_str}")
