import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_cd_file
from edmine.data.FileManager import FileManager
from edmine.model.module.Cluster import HyperCDDeepCluster
from edmine.utils.use_torch import is_cuda_available, is_mps_available


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="ncd_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    graph_dir = os.path.join(setting_dir, "HyperCD")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    H_path = os.path.join(graph_dir, f"user_hyper_graph_{params['train_file_name'].replace('txt', 'npy')}")
    if not os.path.exists(H_path):
        train_data_path = os.path.join(setting_dir, params["train_file_name"])
        data_statics_path = os.path.join(setting_dir, f"{params['dataset_name']}_statics.txt")
        with open(data_statics_path, "r") as f:
            s = f.readline()
            num_user = int(s.split(":")[1].strip())

        q_table = file_manager.get_q_table(params['dataset_name'])
        num_question = q_table.shape[0]
        train_data = read_cd_file(train_data_path)
        r_matrix = -1 * np.ones(shape=(num_user, num_question))
        for interaction in train_data:
            q_id = interaction['question_id']
            user_id = interaction['user_id']
            r_matrix[user_id, q_id] = int(interaction["correctness"])

        # train cluster
        if is_cuda_available():
            device = "cuda"
        elif is_mps_available():
            device = "mps"
        else:
            device = "cpu"
        X = torch.tensor(r_matrix, dtype=torch.float64).to(device)
        n_clusters = int(num_user * 0.02)
        data_loader = DataLoader(dataset=X, batch_size=256, shuffle=False)
        clf = HyperCDDeepCluster(input_dim=num_question,
                                hidden_dims=[512, 256, 128],
                                latent_dim=64,
                                n_clusters=n_clusters).to(device)
        clf.pretrain(data_loader)
        clf.fit(data_loader)
        groups = clf.gain_clusters(data_loader, n_clusters // 2)

        # get hyper graph
        H = np.zeros((num_user, n_clusters))
        for i in range(H.shape[0]):
            H[i, groups[i]] = 1
        H = H[:, np.count_nonzero(H, axis=0) >= 2]  # remove empty edge
        np.save(H_path, H)


        