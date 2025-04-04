import argparse
import os
import torch
import numpy as np
from scipy.sparse import diags

from config import FILE_MANAGER_ROOT

from edmine.data.FileManager import FileManager
from edmine.utils.parse import c2q_from_q_table
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009-single-concept")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    abqr_dir = os.path.join(setting_dir, "ABQR")
    if not os.path.exists(abqr_dir):
        os.mkdir(abqr_dir)
    dataset_name = params["dataset_name"]
    save_path = os.path.join(abqr_dir, f"abqr_graph_{dataset_name}.pt")
    q_table = file_manager.get_q_table(dataset_name)
    c2q = c2q_from_q_table(q_table)
    num_question = q_table.shape[0]
    
    A = np.zeros((num_question, num_question))
    for q_ids in c2q.values():
        for i, q_i in enumerate(q_ids):
            for q_j in q_ids[i:]:
                A[q_i][q_j] = 1
                A[q_j][q_i] = 1
    A = A + np.eye(num_question)
                
    # 计算度矩阵 D（对角线为节点度数）
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = diags(1 / np.sqrt(degrees), offsets=0)  
    # 对称归一化
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    graph = torch.from_numpy(A_normalized).to_sparse_coo()
    torch.save(graph, save_path)
        