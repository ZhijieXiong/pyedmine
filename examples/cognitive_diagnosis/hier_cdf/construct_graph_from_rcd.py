import argparse
import os
import pandas as pd
# import numpy as np
# from scipy.stats import mannwhitneyu
# from networkx import DiGraph, simple_cycles

from config import FILE_MANAGER_ROOT

from edmine.data.FileManager import FileManager
# from edmine.utils.data_io import read_cd_file
# from edmine.utils.parse import c2q_from_q_table


def build_graph4rcd(g_path):
    edge_list = []
    with open(g_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '').split('\t')
            edge_list.append((int(line[0]), int(line[1])))
    return edge_list


def remove_cycles(edges):
    from collections import defaultdict

    # 构建图的邻接表
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    # DFS 检测环路
    visited = set()
    on_stack = set()
    cycle_edges = set()

    def dfs(u):
        visited.add(u)
        on_stack.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
            elif v in on_stack:
                cycle_edges.add((u, v))
        on_stack.remove(u)
    
    for u in list(graph):
        if u not in visited:
            dfs(u)
    
    # 去除环路中的边
    if cycle_edges:
        edges = [edge for edge in edges if edge not in cycle_edges]
    
    return edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="ncd_setting")
    parser.add_argument("--dataset_name", type=str, default="statics2011")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    hcd_dir = os.path.join(setting_dir, "HierCDF")
    if not os.path.exists(hcd_dir):
        os.mkdir(hcd_dir)
    hier_path = os.path.join(hcd_dir, f"{params['dataset_name']}_hier.csv")
    if not os.path.exists(hier_path):
        rcd_dir = os.path.join(setting_dir, "RCD")
        directed_path = os.path.join(rcd_dir, f"{params['dataset_name']}_K_Directed.txt")
        pd.DataFrame(remove_cycles(build_graph4rcd(directed_path)), columns=["from", "to"]).to_csv(hier_path, index=False)
    
        
    # 自己构建
    # q_table = file_manager.get_q_table(params["dataset_name"])
    # c2q = c2q_from_q_table(q_table)
    # num_question, num_concept = q_table.shape[0], q_table.shape[1]
    # hier_path = os.path.join(hcd_dir, f'hier_{params["train_file_name"].replace(".txt", "")}.csv')
    # cd_data = read_cd_file(os.path.join(setting_dir, params["train_file_name"]))
    # data_statics_path = os.path.join(setting_dir, f"{params['dataset_name']}_statics.txt")
    # with open(data_statics_path, "r") as f:
    #     s = f.readline()
    #     num_user = int(s.split(":")[1].strip())
    
    # # 1. 加载数据
    # UQ_mat = np.zeros((num_user, num_question))
    # for interaction in cd_data:
    #     UQ_mat[interaction["user_id"]][interaction["question_id"]] = 1 

    # # 2. 计算知识点掌握状态
    # theta = 0.5
    # mastery = np.zeros((num_user, num_concept))
    # for c_id in range(num_concept):
    #     q_ids = c2q[c_id]
    #     for n in range(num_user):
    #         correct_rate = UQ_mat[n, q_ids].mean()
    #         mastery[n][c_id] = 1 if correct_rate >= theta else 0

    # # 3. 统计依赖关系（Wilcoxon检验）
    # # 使用 Bonferroni 校正 调整显著性水平
    # alpha = 0.05 / (num_concept * (num_concept - 1))
    # edges = []
    # for j in range(num_concept):
    #     for k in range(num_concept):
    #         if j == k:
    #             continue
    #         group1 = mastery[mastery[:, j] == 1][:, k]
    #         group2 = mastery[mastery[:, j] == 0][:, k]
    #         if len(group1) < 5 or len(group2) < 5:
    #             continue
    #         stat, p = mannwhitneyu(group1, group2, alternative='greater')
    #         if p < alpha:
    #             edges.append((j, k))

    # # 4. 构建DAG（示例：简单去环）
    # # 创建有向图
    # graph = DiGraph(edges)
    # # 检测环路
    # cycles = list(simple_cycles(graph))
    # # 去除环路
    # for cycle in cycles:
    #     for i in range(len(cycle)):
    #         edge = (cycle[i], cycle[(i + 1) % len(cycle)])
    #         if edge in edges:
    #             edges.remove(edge)

    # # 5. 导出层次结构
    # hierarchy = pd.DataFrame(edges, columns=["from", "to"])
    # hierarchy.to_csv(hier_path, index=False)
    