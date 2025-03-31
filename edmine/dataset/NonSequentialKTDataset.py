import os
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset

from edmine.utils.data_io import read_kt_file


class DyGKTDataset(Dataset):
    def __init__(self, dataset_config, objects):
        super(DyGKTDataset, self).__init__()
        self.dataset_config = dataset_config
        self.objects = objects
        self.dataset_original = None
        self.dataset_converted = {
            "idx": [],
            "src_node": [],
            "dst_node": [],
            "time": [],
            "correctness": [],
            "user_his_seq": [],
            "que_his_seq": []
        }
        self.dataset = None
        self.process_dataset()
        
    def __len__(self):
        return len(self.dataset["src_node"])

    def __getitem__(self, index):
        result = dict()
        num_neighbor = self.dataset_config["num_neighbor"]
        for key in self.dataset_converted.keys():
            if key in ["user_his_seq", "que_his_seq"]:
                key_data = self.dataset_converted[key][index]
                padding = [0] * (num_neighbor - len(key_data))
                neighbor_idx = torch.tensor(key_data + padding).long().to(self.dataset_config["device"])
                neighbor_time = self.dataset["time"][neighbor_idx]
                neighbor_mask = torch.tensor(
                    [1] * len (key_data) + [0] * len(padding)
                ).long().to(self.dataset_config["device"])
                if key == "user_his_seq":
                    result["neighbor_src_node"] = self.dataset["src_node"][neighbor_idx]
                    result["neighbor_src_time"] = neighbor_time
                    result["neighbor_src_edge"] = neighbor_idx
                    result["neighbor_src_mask"] = neighbor_mask
                else:
                    result["neighbor_dst_node"] = self.dataset["dst_node"][neighbor_idx]
                    result["neighbor_dst_time"] = neighbor_time
                    result["neighbor_dst_edge"] = neighbor_idx
                    result["neighbor_dst_mask"] = neighbor_mask
            else:
                result[key] = self.dataset[key][index]
        return result
        
    def process_dataset(self):
        self.load_dataset()
        self.convert_dataset()
        self.dataset2tensor()

    def load_dataset(self):
        setting_name = self.dataset_config["setting_name"]
        file_name = self.dataset_config["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        self.dataset_original = read_kt_file(dataset_path)
        
    def convert_dataset(self):
        num_neighbor = self.dataset_config["num_neighbor"]
        n = 0
        que_his_seqs = {}
        for user_data in self.dataset_original:
            user_id = user_data["user_id"]
            seq_len = user_data["seq_len"]
            question_seq = user_data["question_seq"][:seq_len]
            correctness_seq = user_data["correctness_seq"][:seq_len]
            time_seq = user_data["time_seq"][:seq_len]
            for i, (q, t, c) in enumerate(zip(question_seq, time_seq, correctness_seq)):
                if q not in que_his_seqs:
                    que_his_seqs[q] = []
                self.dataset_converted["idx"].append(n)
                self.dataset_converted["src_node"].append(user_id)
                self.dataset_converted["dst_node"].append(q)
                self.dataset_converted["time"].append(t)
                self.dataset_converted["correctness"].append(c)
                que_his_seqs[q].append((n, t))
                self.dataset_converted["user_his_seq"].append(
                    list(range(n-i, n))
                    if i < num_neighbor else 
                    list(range(n-num_neighbor, n))
                )
                self.dataset_converted["que_his_seq"].append(None)
                n += 1
        for q, q_his_seqs in que_his_seqs.items():
            q_his_seqs = list(map(lambda x: x[0], sorted(q_his_seqs, key=lambda x: x[1])))
            for i, idx in enumerate(q_his_seqs):
                self.dataset_converted["que_his_seq"][idx] = q_his_seqs[:i] \
                    if i < num_neighbor else \
                        q_his_seqs[i-num_neighbor:]
    
    def dataset2tensor(self):
        self.dataset = {}
        for k in self.dataset_converted.keys():
            if k not in ["user_his_seq", "que_his_seq"]:
                self.dataset[k] = torch.tensor(self.dataset_converted[k]).long().to(self.dataset_config["device"])
    