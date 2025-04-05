import argparse
import os
import numpy as np

from config import FILE_MANAGER_ROOT

from edmine.data.FileManager import FileManager
from edmine.utils.parse import c2q_from_q_table, q2c_from_q_table


def gen_gikt_graph(question2concept, concept2question, q_neighbor_size, c_neighbor_size):
    num_question = len(question2concept)
    num_concept = len(concept2question)
    q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32)
    c_neighbors = np.zeros([num_concept, c_neighbor_size], dtype=np.int32)
    for q_id, neighbors in question2concept.items():
        if len(neighbors) >= q_neighbor_size:
            q_neighbors[q_id] = np.random.choice(neighbors, q_neighbor_size, replace=False)
        else:
            q_neighbors[q_id] = np.random.choice(neighbors, q_neighbor_size, replace=True)
    for c_id, neighbors in concept2question.items():
        if len(neighbors) >= c_neighbor_size:
            c_neighbors[c_id] = np.random.choice(neighbors, c_neighbor_size, replace=False)
        else:
            c_neighbors[c_id] = np.random.choice(neighbors, c_neighbor_size, replace=True)
    return q_neighbors, c_neighbors
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    gikt_dir = os.path.join(setting_dir, "GIKT")
    if not os.path.exists(gikt_dir):
        os.mkdir(gikt_dir)
    dataset_name = params["dataset_name"]
    question_neighbors_path = os.path.join(gikt_dir, f"gikt_question_neighbors_{dataset_name}.npy")
    concept_neighbors_path = os.path.join(gikt_dir, f"gikt_concept_neighbors_{dataset_name}.npy")
    
    q_table = file_manager.get_q_table(dataset_name)
    c2q = c2q_from_q_table(q_table)
    q2c = q2c_from_q_table(q_table)
    num_max_concept = int(q_table.sum(axis=1).max())
    num_q, num_c = q_table.shape[0], q_table.shape[1]
    
    question_neighbors, concept_neighbors = gen_gikt_graph(q2c, c2q, num_max_concept, min(20, int(num_q / num_c)))
    np.save(question_neighbors_path, question_neighbors)
    np.save(concept_neighbors_path, concept_neighbors)
        