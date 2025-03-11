import argparse
import os
import numpy as np

import config
from utils import delete_test_data

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import cal_qc_acc4kt_data, kt_data2user_question_matrix, kt_data2user_concept_matrix, q2c_from_q_table
from edmine.utils.calculate import cosine_similarity_matrix, pearson_similarity


# 基于user-question交互矩阵和user-concept交互矩阵计算用户相似度，用于习题推荐
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    # 构建相似度矩阵的方法
    parser.add_argument("--similarity", type=str, default="cossim", choices=("cossim", "pearson_corr"))
    parser.add_argument("--alpha", type=float, default=1,
                        help="相似度（余弦相似度或者皮尔逊相关系数）的权重")
    parser.add_argument("--beta", type=float, default=1,
                        help="习题知识点相似度（tf-idf）的权重")
    parser.add_argument("--gamma", type=float, default=1,
                        help="难度相似度的权重")
    # 其它
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    params = vars(args)
    np.random.seed(params["seed"])

    setting_name = params["setting_name"]
    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(setting_name)
    save_dir = os.path.join(setting_dir, "user_smi_mat")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    save_path = os.path.join(save_dir, f"{params['dataset_name']}_{params['similarity']}_{alpha}_{beta}_{gamma}.npy")
    
    if not os.path.exists(save_path):
        Q_table = file_manager.get_q_table(params["dataset_name"])
        num_question, num_concept = Q_table.shape[0], Q_table.shape[1]
        q2c = q2c_from_q_table(Q_table)

        users_data = read_kt_file(os.path.join(setting_dir, f"{params['dataset_name']}_user_data.txt"))
        delete_test_data(users_data)
        # 使用知识追踪训练集和验证集的数据找相似用户
        kt_setting_dir = file_manager.get_setting_dir("pykt_setting")
        kt_train_data = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_train.txt"))
        kt_valid_data = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_valid.txt"))
        users_data += kt_train_data + kt_valid_data

        question_acc = cal_qc_acc4kt_data(users_data, "question", 0)
        average_que_acc = sum(question_acc.values()) / len(question_acc)
        question_diff = {}
        for q_id in range(num_question):
            if q_id not in question_acc:
                question_diff[q_id] = average_que_acc
            else:
                question_diff[q_id] = 1 - question_acc[q_id]

        user_average_diff = {}
        for item_data in users_data:
            user_id = item_data["user_id"]
            question_seq = item_data["question_seq"][:item_data["seq_len"]]
            diff_sum = 0
            for q_id in question_seq:
                diff_sum += question_diff[q_id]
            user_average_diff[user_id] = diff_sum / item_data["seq_len"]

        user_ids = list(map(lambda x: x["user_id"], users_data))
        num_user = max(user_ids) + 1
        user_diff_similarity = np.zeros((num_user, num_user))
        for i in user_ids:
            for j in user_ids:
                user_diff_similarity[i][j] = 1 - abs(user_average_diff[i] - user_average_diff[j])

        # user_question_matrix_和user_concept_matrix的行和user_id没关系
        user_question_matrix_ = kt_data2user_question_matrix(users_data, num_question, 0)
        user_concept_matrix_ = kt_data2user_concept_matrix(users_data, num_concept, q2c, 0)
        user_question_matrix = np.zeros((num_user, num_question))
        user_concept_matrix = np.zeros((num_user, num_concept))
        for i, user_data in enumerate(users_data):
            user_id = user_data["user_id"]
            user_question_matrix[user_id] = user_question_matrix_[i]
            user_concept_matrix[user_id] = user_concept_matrix_[i]
            
        if params["similarity"] == "cossim":
            user_que_similarity = cosine_similarity_matrix(user_question_matrix, axis=1)
            user_concept_similarity = cosine_similarity_matrix(user_concept_matrix, axis=1)
        elif params["similarity"] == "pearson_corr":
            user_que_similarity = np.zeros((num_user, num_user))
            user_concept_similarity = np.zeros((num_user, num_user))
            for i in range(num_user):
                for j in range(num_user):
                    si = user_question_matrix[i, :]
                    sj = user_question_matrix[j, :]
                    user_que_similarity[i][j] = pearson_similarity(si, sj)
            for i in range(num_user):
                for j in range(num_user):
                    si = user_concept_matrix[i, :]
                    sj = user_concept_matrix[j, :]
                    user_concept_similarity[i][j] = pearson_similarity(si, sj)
        else:
            raise NotImplementedError(f'{params["similarity"]} is not implemented')
        user_sim_matrix = alpha * user_que_similarity + beta * user_concept_similarity + gamma * user_diff_similarity
        np.save(save_path, user_sim_matrix)
