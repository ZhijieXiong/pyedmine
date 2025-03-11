import argparse
import os
import numpy as np

import config
from utils import delete_test_data

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import cal_qc_acc4kt_data, kt_data2user_question_matrix, str2bool
from edmine.utils.calculate import tf_idf_from_q_table, cosine_similarity_matrix, pearson_similarity


# 基于user-question的交互矩阵和question-concept的相关矩阵计算习题相似度，用于习题推荐
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据集相关
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    # 是否使用知识追踪的训练集和验证集（假设为ER_offline_setting的话）
    parser.add_argument("--use_kt_train_valid", type=str2bool, default=True)
    # 构建相似度矩阵的方法
    parser.add_argument("--similarity", type=str, default="cossim", choices=("cossim", "pearson_corr"))
    parser.add_argument("--alpha", type=float, default=1,
                        help="相似度（余弦相似度或者皮尔逊相关系数）的权重")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="习题知识点相似度（tf-idf）的权重")
    parser.add_argument("--gamma", type=float, default=0.5,
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
    save_path = os.path.join(setting_dir, f"{params['dataset_name']}_{params['similarity']}_{alpha}_{beta}_{gamma}.npy")
    
    if not os.path.exists(save_path):
        # 这里假设了是ER_offline_setting，则构建user-question交互矩阵时，只使用知识追踪测试集中属于习题推荐训练集和验证集部分的数据
        # 也可以在这里把知识追踪实验的训练集和验证集加上
        users_data = read_kt_file(os.path.join(setting_dir, f"{params['dataset_name']}_user_data.txt"))
        delete_test_data(users_data)
        if params["use_kt_train_valid"]:
            kt_setting_dir = file_manager.get_setting_dir("pykt_setting")
            kt_train_data = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_train.txt"))
            kt_valid_data = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_valid.txt"))
            users_data += kt_train_data + kt_valid_data

        Q_table = file_manager.get_q_table(params["dataset_name"])
        num_question, num_concept = Q_table.shape[0], Q_table.shape[1]

        question_acc = cal_qc_acc4kt_data(users_data, "question", 0)
        average_acc = sum(question_acc.values()) / len(question_acc)
        question_diff = {}
        for q_id in range(num_question):
            if q_id not in question_acc:
                question_diff[q_id] = average_acc
            else:
                question_diff[q_id] = 1 - question_acc[q_id]
        difficulty_dissimilarity = np.zeros((num_question, num_question))
        for i in range(num_question):
            for j in range(num_question):
                difficulty_dissimilarity[i][j] = abs(question_diff[i] - question_diff[j])

        tf_idf = tf_idf_from_q_table(Q_table)
        concept_similarity = cosine_similarity_matrix(Q_table, axis=1)

        user_question_matrix = kt_data2user_question_matrix(users_data, num_question, 0)
        if params["similarity"] == "cossim":
            similarity = cosine_similarity_matrix(user_question_matrix, axis=0)
        elif params["similarity"] == "pearson_corr":
            similarity = np.zeros((num_question, num_question))
            for i in range(num_question):
                for j in range(num_question):
                    si = user_question_matrix[:, i]
                    sj = user_question_matrix[:, j]
                    similarity[i][j] = pearson_similarity(si, sj)
        else:
            raise NotImplementedError(f'{params["similarity"]} is not implemented')
        
        que_sim_matrix = alpha * similarity + beta * concept_similarity - gamma * difficulty_dissimilarity
        np.save(save_path, que_sim_matrix)
