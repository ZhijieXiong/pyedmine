import argparse
import os
import json
import numpy as np

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


def RCD_construct_dependency_matrix(kt_data, q_table):
    """
    RCD所使用的构造知识点关联矩阵的方法
    :param data_uniformed:
    :param num_concept:
    :param Q_table:
    :return:
    """
    edge_dic_deno = {}
    num_concept = q_table.shape[1]
    q2c = q2c_from_q_table(q_table)
    
    # Calculate correct matrix
    concept_correct = np.zeros([num_concept, num_concept])
    for item_data in kt_data:
        if item_data["seq_len"] < 3:
            continue

        for log_i in range(item_data["seq_len"] - 1):
            if item_data["correctness_seq"][log_i] * item_data["correctness_seq"][log_i+1] == 1:
                current_cs = q2c[item_data["question_seq"][log_i]]
                next_cs = q2c[item_data["question_seq"][log_i + 1]]
                for ci in current_cs:
                    for cj in next_cs:
                        if ci != cj:
                            concept_correct[ci][cj] += 1.0
                            # calculate the number of correctly answering i
                            edge_dic_deno.setdefault(ci, 1)
                            edge_dic_deno[ci] += 1

    s = 0
    c = 0
    # Calculate transition matrix
    concept_directed = np.zeros([num_concept, num_concept])
    for i in range(num_concept):
        for j in range(num_concept):
            if i != j and concept_correct[i][j] > 0:
                concept_directed[i][j] = float(concept_correct[i][j]) / edge_dic_deno[i]
                s += concept_directed[i][j]
                c += 1
    o = np.zeros([num_concept, num_concept])
    min_c = 100000
    max_c = 0
    for i in range(num_concept):
        for j in range(num_concept):
            if concept_correct[i][j] > 0 and i != j:
                min_c = min(min_c, concept_directed[i][j])
                max_c = max(max_c, concept_directed[i][j])
    s_o = 0
    l_o = 0
    for i in range(num_concept):
        for j in range(num_concept):
            if concept_correct[i][j] > 0 and i != j:
                o[i][j] = (concept_directed[i][j] - min_c) / (max_c - min_c)
                l_o += 1
                s_o += o[i][j]

    # avg^2 is threshold
    threshold = (s_o / l_o) ** 2
    graph = ''
    for i in range(num_concept):
        for j in range(num_concept):
            if o[i][j] >= threshold:
                graph += str(i) + '\t' + str(j) + '\n'

    return graph
        
        
# 理论上只能用训练集构建graph，但是CD data是无序的，只能用全部数据来构建
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="ncd_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    graph_dir = os.path.join(setting_dir, "RCD")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    data_path = file_manager.get_preprocessed_path(params["dataset_name"])
    directed_path = os.path.join(graph_dir, f'{params["dataset_name"]}_K_Directed.txt')
    undirected_path = os.path.join(graph_dir, f'{params["dataset_name"]}_K_Undirected.txt')
    
    if (not os.path.exists(directed_path)) or (not os.path.join(undirected_path)):
        kt_data_ = read_kt_file(data_path)
        q_table_ = file_manager.get_q_table(params['dataset_name'])
        graph_ = RCD_construct_dependency_matrix(kt_data_, q_table_).strip()
        relations = graph_.split("\n")
        K_Directed = ''
        K_Undirected = ''
        edge = []
        for relation in relations:
            relation = relation.replace('\n', '').split('\t')
            src = relation[0]
            tar = relation[1]
            edge.append((src, tar))
        visit = []
        for e in edge:
            if e not in visit:
                if (e[1],e[0]) in edge:
                    K_Undirected += str(e[0] + '\t' + e[1] + '\n')
                    visit.append(e)
                    visit.append((e[1],e[0]))
                else:
                    K_Directed += str(e[0] + '\t' + e[1] + '\n')
                    visit.append(e)
        with open(directed_path, 'w') as f:
            f.write(K_Directed)
        with open(undirected_path, 'w') as f:
            f.write(K_Undirected)
            