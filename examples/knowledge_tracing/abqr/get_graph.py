import argparse
import os
import torch
import numpy as np
from scipy.sparse import lil_matrix, diags

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
    
    # 加载 q_table 和 c2q
    q_table = file_manager.get_q_table(dataset_name)
    c2q = c2q_from_q_table(q_table)
    num_question = q_table.shape[0]
    
    # 使用稀疏矩阵 LIL 格式（适合逐步填充）
    A = lil_matrix((num_question, num_question), dtype=np.float32)  # 节省内存
    
    # 填充邻接矩阵（仅存储非零元素）
    for q_ids in c2q.values():
        for i, q_i in enumerate(q_ids):
            for q_j in q_ids[i:]:
                A[q_i, q_j] = 1
                A[q_j, q_i] = 1
    
    # 添加自环（对角线元素）
    A.setdiag(1)  # 直接操作稀疏矩阵的对角线
    
    # 计算度矩阵 D 和对称归一化
    degrees = np.array(A.sum(axis=1)).flatten()  # 度数为稠密数组
    D_inv_sqrt = diags(1 / np.sqrt(degrees), offsets=0, format="csr")  # CSR 格式高效计算
    
    # 稀疏矩阵乘法（避免中间稠密结果）
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
    
    # 转换为 PyTorch 稀疏张量
    A_normalized = A_normalized.tocoo()  # 转为 COO 格式
    indices = torch.stack([torch.tensor(A_normalized.row), torch.tensor(A_normalized.col)])
    values = torch.tensor(A_normalized.data)
    graph = torch.sparse_coo_tensor(indices, values, A_normalized.shape)
    
    torch.save(graph, save_path)