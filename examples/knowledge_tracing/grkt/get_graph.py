import argparse
import os
import numpy as np
from tqdm import tqdm

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


def get_graph(kt_data_, num_c, q2c_):
    tt = np.zeros((num_c + 1, num_c + 1))
    tf = np.zeros((num_c + 1, num_c + 1))
    ft = np.zeros((num_c + 1, num_c + 1))
    ff = np.zeros((num_c + 1, num_c + 1))

    for user_data in tqdm(kt_data_):
        t = np.zeros(num_c + 1)
        f = np.zeros(num_c + 1)
        seq_len = user_data["seq_len"]
        question_seq = user_data["question_seq"][:seq_len]
        correctness_seq = user_data["correctness_seq"][:seq_len]
        for q_id, corr in zip(question_seq, correctness_seq):
            know = q2c_[q_id][0]
            if corr:
                tt[:, know] += t
                tt[know, :] += t
                ft[:, know] += f
                tf[know, :] += f
            else:
                ff[:, know] += f
                ff[know, :] += f
                tf[:, know] += t
                ft[know, :] += t
            if corr:
                t[know] += 1
            else:
                f[know] += 1
    cold_thresh = 5
    rel_filt = (tt + ff + tf + ft) >= 4*cold_thresh
    pre_filt = (tf + ft) >= 2*cold_thresh

    rel_map_ = (tt + ff)/np.clip(tt + ff + tf + ft, a_min = 1, a_max = None)
    pre_map_ = ft/np.clip(tf + ft, a_min = 1, a_max = None)

    for i in range(len(rel_map_)):
        rel_map_[i, i] = 0
        pre_map_[i, i] = 0

    rel_map_ = rel_map_*rel_filt
    pre_map_ = pre_map_*pre_filt
    
    return rel_map_, pre_map_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    lbkt_dir = os.path.join(setting_dir, "GRKT")
    if not os.path.exists(lbkt_dir):
        os.mkdir(lbkt_dir)
    train_file_name = params["train_file_name"]
    rel_map_path = os.path.join(lbkt_dir, train_file_name.replace(".txt", "_grkt_rel_map.npy"))
    pre_map_path = os.path.join(lbkt_dir, train_file_name.replace(".txt", "_grkt_pre_map.npy"))
    if not (os.path.exists(rel_map_path) and os.path.exists(pre_map_path)):
        kt_data = read_kt_file(os.path.join(setting_dir, train_file_name))
        q_table = file_manager.get_q_table(params["dataset_name"])
        q2c = q2c_from_q_table(q_table)
        num_concept = q_table.shape[1]
        rel_map, pre_map = get_graph(kt_data, num_concept, q2c)
        np.save(rel_map_path, rel_map)
        np.save(pre_map_path, pre_map)
        