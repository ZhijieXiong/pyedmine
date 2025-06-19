import os
import torch
from torch.utils.data import Dataset

from edmine.utils.data_io import read_kt_file


class DyGKTDataset(Dataset):
    def __init__(self, dataset_config, objects, is_train=True):
        super(DyGKTDataset, self).__init__()
        self.dataset_config = dataset_config
        self.objects = objects
        self.train_dataset = None
        self.test_dataset = None
        self.is_train = is_train
        self.dataset_converted = {
            "idx": [],
            "user": [],
            "question": [],
            "idx_in_seq": [],
            "time": [],
            "correctness": [],
            "user_his_seq": [],
            "user_his_snq_seq": [],
            "user_his_snk_seq": [],
            "que_his_seq": [],
            "que_his_qn_seq": [],
            "is_train": []
        }
        self.dataset = None
        self.process_dataset()
        
    def __len__(self):
        return self.dataset["user"].shape[0]
        
    def __getitem__(self, index):
        result = dict()
        num_neighbor = self.dataset_config["num_neighbor"]
        for key in self.dataset_converted.keys():
            if key in ["user_his_seq", "que_his_seq"]:
                key_data = self.dataset_converted[key][index]
                padding = [0] * (num_neighbor - len(key_data))
                neighbor_idx = torch.tensor(key_data + padding).long().to(self.dataset_config["device"])
                neighbor_time = self.dataset["time"][neighbor_idx]
                neighbor_edge = self.dataset["correctness"][neighbor_idx]
                neighbor_last_idx = torch.tensor(len(key_data)).long().to(self.dataset_config["device"])
                if key == "user_his_seq":
                    result["user_his_time_seq"] = neighbor_time
                    result["user_his_correctness_seq"] = neighbor_edge
                    result["user_his_last_idx"] = neighbor_last_idx
                else:
                    result["que_his_time_seq"] = neighbor_time
                    result["que_his_correctness_seq"] = neighbor_edge
                    result["que_his_last_idx"] = neighbor_last_idx
            elif key in ["user_his_snq_seq", "user_his_snk_seq", "que_his_qn_seq"]:
                key_data = self.dataset_converted[key][index]
                padding = [0] * (num_neighbor - len(key_data))
                result[key] = torch.tensor(key_data + padding).long().to(self.dataset_config["device"])
            else:
                result[key] = self.dataset[key][index]
        return result
        
    def process_dataset(self):
        self.load_dataset()
        self.convert_dataset()
        self.dataset2tensor()
        
    def read_kt_file(self, file_name):
        setting_name = self.dataset_config["setting_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        return read_kt_file(dataset_path)

    def load_dataset(self):
        self.data_all = []
        for file_name in self.dataset_config["file_names"]:
            self.data_all = self.read_kt_file(file_name)
        
    def convert_dataset(self):
        q_table = self.objects["dataset"]["q_table"]
        que_sim_by_concept = ((q_table @ q_table.T) > 0).astype(int)
        num_question = self.dataset_config["num_question"]
        num_neighbor = self.dataset_config["num_neighbor"]
        n = 0
        que_his_seqs = {}
        for user_data in self.data_all:
            user_id = num_question + user_data["user_id"]
            seq_len = user_data["seq_len"]
            question_seq = user_data["question_seq"][:seq_len]
            correctness_seq = user_data["correctness_seq"][:seq_len]
            time_seq = user_data["time_seq"][:seq_len]
            for i, (q_id, t, c) in enumerate(zip(question_seq, time_seq, correctness_seq)):
                if q_id not in que_his_seqs:
                    que_his_seqs[q_id] = []
                que_his_seqs[q_id].append((n, t))
                self.dataset_converted["idx"].append(n)
                self.dataset_converted["user"].append(user_id)
                self.dataset_converted["question"].append(q_id)
                self.dataset_converted["idx_in_seq"].append(i)
                self.dataset_converted["time"].append(t)
                self.dataset_converted["correctness"].append(c)
                user_his_seq = list(range(n-i, n)) if i < num_neighbor else list(range(n-num_neighbor, n))
                self.dataset_converted["user_his_seq"].append(user_his_seq)
                question_seq_ = question_seq[0 if (i <= num_neighbor) else (i-num_neighbor):i]
                user_his_snd_seq = list(map(lambda q: int(q == q_id), question_seq_))
                user_his_snk_seq = list(map(lambda q: int(que_sim_by_concept[q, q_id]), question_seq_))
                self.dataset_converted["user_his_snd_seq"].append(user_his_snd_seq)
                self.dataset_converted["user_his_snk_seq"].append(user_his_snk_seq)
                self.dataset_converted["que_his_seq"].append(None)
                self.dataset_converted["que_his_qn_seq"].append(None)
                n += 1
                
        for i in range(n):
            q_id = self.dataset_converted["question"][i]
            t = self.dataset_converted["time"][i]
            que_his_seq = list(map(
                lambda x: x[0], sorted(
                    list(filter(
                        lambda y: y[1] < t, que_his_seqs[q_id]
                    )), key=lambda z: z[1]
                )
            ))
            self.dataset_converted["que_his_seq"][i] = \
                que_his_seq if len(que_his_seq) < num_neighbor else que_his_seq[-num_neighbor:]

    
    def dataset2tensor(self):
        self.dataset = {}
        for k in self.dataset_converted.keys():
            if k not in ["user_his_seq", "que_his_seq"]:
                self.dataset[k] = torch.tensor(self.dataset_converted[k]).long().to(self.dataset_config["device"])
    