import argparse
import os
import numpy as np
from collections import defaultdict

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager


def build_direct_support_graph(kt_data, num_question, omega):
    # 统计问题共现计数
    count_matrix = defaultdict(lambda: defaultdict(lambda: {'11': 0, '10': 0, '01': 0, '00': 0}))
    total_counts = defaultdict(int)
    
    # 遍历所有学习记录
    for record in kt_data:
        q_seq = record["question_seq"]
        c_seq = record["correctness_seq"]
        seq_len = record["seq_len"]
        
        # 仅处理有效序列部分
        for i in range(seq_len-1):
            for j in range(i+1, seq_len):
                e1, r1 = q_seq[i], c_seq[i]
                e2, r2 = q_seq[j], c_seq[j]
                key = f"{r1}{r2}"
                count_matrix[e1][e2][key] += 1
                total_counts[(e1, e2)] += 1

    # 构建邻接矩阵
    A_e = np.zeros((num_question, num_question), dtype=np.float32)
    for e1 in count_matrix:
        for e2 in count_matrix[e1]:
            # 公式3
            P_R1_condi_R2_nume = count_matrix[e1][e2]['11'] + 0.01
            P_R1_condi_R2_deno = (count_matrix[e1][e2]['11'] + count_matrix[e1][e2]['01']) + 0.01
            P_R1_condi_R2 = P_R1_condi_R2_nume / P_R1_condi_R2_deno
            
            # 公式4
            P_R1_condi_RW2_nume = count_matrix[e1][e2]['10'] + count_matrix[e1][e2]['11'] + 0.01
            P_R1_condi_RW2_deno = total_counts[(e1,e2)] + 0.01
            P_R1_condi_RW2 = P_R1_condi_RW2_nume / P_R1_condi_RW2_deno
            
            # 类比公式3、4
            P_W2_condi_W1 = (count_matrix[e1][e2]['00'] + 0.01) / ((count_matrix[e1][e2]['00'] + count_matrix[e1][e2]['01']) + 0.01)
            P_W2_condi_RW1 = (count_matrix[e1][e2]['10'] + count_matrix[e1][e2]['00'] + 0.01) / (total_counts[(e1,e2)] + 0.01)
            
            support = max(0, np.log(P_R1_condi_R2 / P_R1_condi_RW2)) + \
                max(0, np.log(P_W2_condi_W1 / P_W2_condi_RW1))
            
            if support > omega:
                A_e[e1,e2] = 1
                A_e[e2,e1] = 1  # 无向图

    return A_e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    hgkt_dir = os.path.join(setting_dir, "HGKT")
    if not os.path.exists(hgkt_dir):
        os.mkdir(hgkt_dir)
    train_file_name = params["train_file_name"]
    save_path = os.path.join(hgkt_dir, train_file_name.replace(".txt", "_hgkt_direct_support_graph.txt"))
    q_table = file_manager.get_q_table(params["dataset_name"])
    num_q = q_table.shape[0]
    data = read_kt_file(os.path.join(setting_dir, train_file_name))
    num_interaction = sum(list(map(lambda x: x["seq_len"], data)))
    # 论文使用的数据集是11410道习题
    omega = 2.3
    print(omega)
    A = build_direct_support_graph(data, num_q, omega)
    with open(save_path, "w") as f:
        for i in range(num_q):
            for j in range(i+1, num_q):
                if A[i][j] == 1:
                    f.write(f"{i},{j}\n")