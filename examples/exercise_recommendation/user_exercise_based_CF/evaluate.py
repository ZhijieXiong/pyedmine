import argparse
import json
import os
import numpy as np

import config
from rec_strategy import *

from edmine.utils.data_io import read_mlkc_data, read_kt_file
from edmine.utils.parse import q2c_from_q_table
from edmine.data.FileManager import FileManager
from edmine.metric.exercise_recommendation import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="kg4ex_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    parser.add_argument("--user_data_file_name", type=str, default="assist2009_user_data.txt")
    parser.add_argument("--que_sim_mat_file_name", type=str, default="que_smi_mat_statics2011_pearson_corr_1_0.25_0.5.npy")
    # 评价指标选择
    parser.add_argument("--used_metrics", type=str, default="['KG4EX_ACC', 'KG4EX_NOV', 'PER_IND']",
                        help='KG4EX_ACC, KG4EX_VOL, PER_IND')
    parser.add_argument("--top_ns", type=str, default="[5,10,20]")
    # KG4EX_ACC指标需要的数据
    parser.add_argument("--mlkc_file_name", type=str, default="statics2011_mlkc_test.txt")
    parser.add_argument("--delta", type=float, default=0.7)
    # 推荐策略
    parser.add_argument("--rec_strategy", type=int, default=0,
                        help="0: 推荐和学生最后一次做错习题相似的习题，如果学生历史练习习题全部做对，则推荐和最后练习习题相似的习题")
    # 其它
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    params = vars(args)
    np.random.seed(params["seed"])

    setting_name = params["setting_name"]
    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(setting_name)
    Q_table = file_manager.get_q_table(params["dataset_name"])
    question2concept = q2c_from_q_table(Q_table)
    num_question, num_concept = Q_table.shape[0], Q_table.shape[1]

    users_data = read_kt_file(os.path.join(setting_dir, params["user_data_file_name"]))
    que_sim_mat = np.load(os.path.join(setting_dir, params["que_sim_mat_file_name"]))
    similar_questions = np.argsort(-que_sim_mat, axis=1)[:, 1:]

    rec_strategy = params["rec_strategy"]
    top_ns = eval(params["top_ns"])
    rec_result = {x: {} for x in top_ns}
    if rec_strategy == 0:
        top_ns = sorted(top_ns, reverse=True)
        last_top_n = top_ns[0]
        for i, top_n in enumerate(top_ns):
            if i == 0:
                rec_result[top_n] = rec_method_0(users_data, similar_questions, top_n)
            else:
                rec_result[top_n] = {
                    user_id: rec_ques[:top_n] for user_id, rec_ques in rec_result[last_top_n].items()}
                last_top_n = top_n
    else:
        raise ValueError(f"{rec_strategy} is not implemented")

    used_metrics = eval(params["used_metrics"])
    performance = {x: {} for x in top_ns}
    for metric in used_metrics:
        for top_n in top_ns:
            rec_ques = rec_result[top_n]
            if metric == "PER_IND":
                rec_ques_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                performance[top_n][metric] = personalization_index(rec_ques_)
            if metric == "KG4EX_ACC":
                mlkc = read_mlkc_data(os.path.join(setting_dir, params["mlkc_file_name"]))
                rec_ques_ = []
                mlkc_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    mlkc_.append(mlkc[user_id])

                performance[top_n][metric] = kg4ex_acc(mlkc_, rec_ques_, question2concept, params["delta"])
            if metric == "KG4EX_NOV":
                correct_cs = {}
                for item_data in users_data:
                    user_id = item_data["user_id"]
                    seq_len = item_data["seq_len"]
                    question_seq = item_data["question_seq"][:seq_len]
                    correct_seq = item_data["correct_seq"][:seq_len]
                    correct_cs[user_id] = get_user_answer_correctly_concepts(question_seq, correct_seq, question2concept)

                rec_ques_ = []
                correct_cs_ = []
                for user_id in rec_ques.keys():
                    rec_ques_.append(rec_ques[user_id])
                    correct_cs_.append(correct_cs[user_id])

                performance[top_n][metric] = kg4ex_novelty(correct_cs_, rec_ques_, question2concept)

    top_ns = sorted(top_ns)
    for top_n in top_ns:
        print(f"top {top_n}, {json.dumps(performance[top_n])}")
