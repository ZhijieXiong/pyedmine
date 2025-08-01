import argparse
import os
from copy import deepcopy

import config

from edmine.data.FileManager import FileManager
from edmine.utils.data_io import read_kt_file, write_kt_file
from edmine.utils.parse import q2c_from_q_table


def get_unique_concepts(question_seq, q2c):
    concepts = []
    for q_id in question_seq:
        concepts += q2c[q_id]
    return set(concepts)


def split_seq(seqs, min_seq_len, split_seq):
    new_seqs = []
    for user_data in seqs:
        seq_len = user_data["seq_len"]
        if seq_len < min_seq_len:
            continue
        user_data["intial_idx"] = int(seq_len * split_seq)
        new_seqs.append(user_data)
    return new_seqs
        
        
def generate_lpr_data(kt_data):
    single_goal_data = []
    multi_goals_data = []
    for user_data in kt_data:
        intial_idx = user_data["intial_idx"]
        del user_data["intial_idx"]
        future_q_seq = user_data["question_seq"][intial_idx:user_data["seq_len"]]
        future_concepts = get_unique_concepts(future_q_seq, question2concept)
        user_data_ = deepcopy(user_data)
        user_data_["seq_len"] = intial_idx
        for k, v in user_data_.items():
            if type(v) is list:
                user_data_[k] = v[:intial_idx]
        user_data_["learning_goals"] = list(future_concepts)
        multi_goals_data.append(user_data_)
        for learning_goal in future_concepts:
            user_data_ = deepcopy(user_data)
            user_data_["learning_goal"] = learning_goal
            user_data_["seq_len"] = intial_idx
            for k, v in user_data_.items():
                if type(v) is list:
                    user_data_[k] = v[:intial_idx]
            single_goal_data.append(user_data_)
    return single_goal_data, multi_goals_data


if __name__ == "__main__":
    # 离线评估使用真实学生的部分数据初始化智能体状态，然后用该学生未来所练习习题对应知识点作为学习目标
    # LPR任务大多数都是用强化学习做的，其本质可以视为Meta-RL，即将学生视为任务
    # 为了检测算法是否有效，所以KTTest的数据用来作为LPRTest数据，确保训练过程没有见过
    # 同理可以使用KTTrain和KTValid来划分LPRTrain和LPRValid
    # 为了减少工作量，基于pykt_setting划分数据集
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)

    lpr_setting = {
        "name": "LPR_offline_setting",
        "min_seq_len": 20,
        "split_seq": 0.8
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(lpr_setting["name"], lpr_setting)
    q_table = file_manager.get_q_table(params['dataset_name'])
    question2concept = q2c_from_q_table(q_table)
    lpr_setting_dir = file_manager.get_setting_dir(lpr_setting["name"])
    kt_setting_dir = file_manager.get_setting_dir("pykt_setting")
    
    kt_data_train = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_train.txt"))
    kt_data_valid = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_valid.txt"))
    kt_data_test = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_test.txt"))
    
    kt_data_train = split_seq(kt_data_train, lpr_setting["min_seq_len"], lpr_setting["split_seq"])
    kt_data_valid = split_seq(kt_data_valid, lpr_setting["min_seq_len"], lpr_setting["split_seq"])
    kt_data_test = split_seq(kt_data_test, lpr_setting["min_seq_len"], lpr_setting["split_seq"])

    lpr_data_train = generate_lpr_data(kt_data_train)
    lpr_data_valid = generate_lpr_data(kt_data_valid)
    lpr_data_test = generate_lpr_data(kt_data_test)
    
    write_kt_file(lpr_data_train[0], os.path.join(lpr_setting_dir, f"{params['dataset_name']}_single_goal_train.txt"))
    write_kt_file(lpr_data_train[1], os.path.join(lpr_setting_dir, f"{params['dataset_name']}_multi_goals_train.txt"))
    write_kt_file(lpr_data_valid[0], os.path.join(lpr_setting_dir, f"{params['dataset_name']}_single_goal_valid.txt"))
    write_kt_file(lpr_data_valid[1], os.path.join(lpr_setting_dir, f"{params['dataset_name']}_multi_goals_valid.txt"))
    write_kt_file(lpr_data_test[0], os.path.join(lpr_setting_dir, f"{params['dataset_name']}_single_goal_test.txt"))
    write_kt_file(lpr_data_test[1], os.path.join(lpr_setting_dir, f"{params['dataset_name']}_multi_goals_test.txt"))
