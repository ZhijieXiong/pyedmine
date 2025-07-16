import argparse
import os
import math
from collections import defaultdict

from config import FILE_MANAGER_ROOT
from edmine.data.FileManager import FileManager
from edmine.utils.data_io import read_kt_file
from edmine.utils.parse import q2c_from_q_table

from utils import check_cycles


# 由于GKT（DLPR论文中说的是使用GKT中提及的方法来提取Graph）所使用的方法边太多
# 于是加了两层过滤：过滤PMI<=0的边和过滤双向边，Assist2009数据集提取出773条边，和论文汇报（683）差不多
def build_prerequisite_edges(user_data, q2c):
    """
    Use PMI to extract prerequisite concept relations and filter out
    mutually directed (bidirectional) edges.

    Args:
        user_data: List of dicts with 'question_seq'
        q2c: Mapping from question ID to list of concept IDs

    Returns:
        List of (c1, c2) where c1 is a likely prerequisite of c2
    """

    # Step 1: count transitions
    concept_pre_count = defaultdict(int)
    concept_post_count = defaultdict(int)
    co_occurrence = defaultdict(int)
    total_transitions = 0

    for record in user_data:
        q_seq = record["question_seq"]
        seq_len = record["seq_len"]
        for i in range(seq_len - 1):
            q1, q2 = q_seq[i], q_seq[i + 1]
            c1_list = q2c.get(q1, [])
            c2_list = q2c.get(q2, [])
            for c1 in c1_list:
                for c2 in c2_list:
                    if c1 != -1 and c2 != -1 and c1 != c2:
                        co_occurrence[(c1, c2)] += 1
                        concept_pre_count[c1] += 1
                        concept_post_count[c2] += 1
                        total_transitions += 1

    # Step 2: compute PMI > 0 edges
    raw_edges = set()
    for (c1, c2), count in co_occurrence.items():
        p_c1 = concept_pre_count[c1] / total_transitions
        p_c2 = concept_post_count[c2] / total_transitions
        p_c1c2 = count / total_transitions

        if p_c1 * p_c2 == 0:
            continue

        pmi = math.log(p_c1c2 / (p_c1 * p_c2), 2)
        if pmi > 0:
            raw_edges.add((c1, c2))

    # Step 3: filter out bidirectional edges
    final_edges = []
    for (c1, c2) in raw_edges:
        if (c2, c1) not in raw_edges:
            final_edges.append((c1, c2))
            
    # check_cycles(final_edges)

    return final_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="LPR_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)
    
    dataset_name = params["dataset_name"]
    setting_name = params["setting_name"]
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    kt_data_path = os.path.join(file_manager.get_preprocessed_path(dataset_name))
    setting_dir = file_manager.get_setting_dir(setting_name)
    kt_data = read_kt_file(kt_data_path)
    q_table = file_manager.get_q_table(dataset_name)
    question2concept = q2c_from_q_table(q_table)
    
    dlpr_dir = os.path.join(setting_dir, "DLPR")
    if not os.path.exists(dlpr_dir):
        os.mkdir(dlpr_dir)
    save_path = os.path.join(dlpr_dir, f"{dataset_name}_pre_relation.txt")
    
    edge_id_pairs = build_prerequisite_edges(kt_data, question2concept)
    with open(save_path, "w") as f:
        for from_id, to_id in sorted(edge_id_pairs):
            f.write(f"{from_id},{to_id}\n")
    