import os
import numpy as np
import torch

from torch.utils.data import Dataset

from edmine.utils.data_io import read_kt_file
from edmine.utils.parse import get_keys_from_kt_data


class BasicSequentialKTDataset(Dataset):
    def __init__(self, dataset_config, objects):
        super(BasicSequentialKTDataset, self).__init__()
        self.dataset_config = dataset_config
        self.objects = objects
        self.dataset_original = None
        self.dataset_converted = None
        self.dataset = None
        self.process_dataset()

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
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
        id_keys, seq_keys = get_keys_from_kt_data(self.dataset_original)
        self.dataset_converted = {k: [] for k in (id_keys + seq_keys)}
        if "time_seq" in seq_keys:
            self.dataset_converted["interval_time_seq"] = []
        max_seq_len = len(self.dataset_original[0]["mask_seq"])
        for _, item_data in enumerate(self.dataset_original):
            seq_len = item_data["seq_len"]
            for k in id_keys:
                self.dataset_converted[k].append(item_data[k])
            for k in seq_keys:
                if k == "time_seq":
                    interval_time_seq = [0]
                    for time_i in range(1, seq_len):
                        interval_time_real = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                        interval_time_idx = max(0, interval_time_real)
                        interval_time_seq.append(interval_time_idx)
                    interval_time_seq += [0] * (max_seq_len - seq_len)
                    self.dataset_converted["interval_time_seq"].append(interval_time_seq)
                else:
                    self.dataset_converted[k].append(item_data[k])

        if "time_seq" in self.dataset_converted.keys():
            del self.dataset_converted["time_seq"]

    def dataset2tensor(self):
        self.dataset = {}
        for k in self.dataset_converted.keys():
            self.dataset[k] = torch.tensor(self.dataset_converted[k]).long().to(self.dataset_config["device"])

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correctness_seq = self.dataset["correctness_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correctness_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction


class DIMKTDataset(BasicSequentialKTDataset):
    def __init__(self, dataset_config, objects):
        super(DIMKTDataset, self).__init__(dataset_config, objects)

    def process_dataset(self):
        self.load_dataset()
        self.parse_difficulty()
        self.convert_dataset()
        self.dataset2tensor()

    def parse_difficulty(self):
        question_difficulty = self.objects["dimkt"]["question_difficulty"]
        for item_data in self.dataset_original:
            item_data["question_diff_seq"] = []
            for q_id in item_data["question_seq"]:
                item_data["question_diff_seq"].append(question_difficulty[q_id])


class QDCKTDataset(BasicSequentialKTDataset):
    def __init__(self, dataset_config, objects, train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            q_table = objects["dataset"]["q_table"]
            self.q_with_same_concepts = {}
            for q_id in range(q_table.shape[0]):
                q_table_ = q_table - np.tile(q_table[q_id], (q_table.shape[0], 1))
                q_table_sum = q_table_.sum(axis=1)
                self.q_with_same_concepts[q_id] = list(set(np.nonzero(q_table_sum == 0)[0]) - {q_id})
        super(QDCKTDataset, self).__init__(dataset_config, objects)

    def process_dataset(self):
        self.load_dataset()
        self.parse_difficulty()
        self.convert_dataset()
        self.dataset2tensor()

    def parse_difficulty(self):
        question_difficulty = self.objects["qdckt"]["question_difficulty"]
        for item_data in self.dataset_original:
            item_data["question_diff_seq"] = []
            item_data["similar_question_seq"] = []
            item_data["similar_question_diff_seq"] = []
            item_data["similar_question_mask_seq"] = []
            for i, q_id in enumerate(item_data["question_seq"]):
                item_data["question_diff_seq"].append(question_difficulty[q_id])
                if self.train_mode:
                    if i >= item_data["seq_len"]:
                        item_data["similar_question_seq"].append(0)
                        item_data["similar_question_diff_seq"].append(0)
                        item_data["similar_question_mask_seq"].append(0)
                    else:
                        similar_q_ids = self.q_with_same_concepts[q_id]
                        if len(similar_q_ids) > 0:
                            similar_q_id = np.random.choice(similar_q_ids)
                            item_data["similar_question_seq"].append(similar_q_id)
                            item_data["similar_question_diff_seq"].append(question_difficulty[similar_q_id])
                            item_data["similar_question_mask_seq"].append(1)
                        else:
                            item_data["similar_question_seq"].append(0)
                            item_data["similar_question_diff_seq"].append(0)
                            item_data["similar_question_mask_seq"].append(0)
            if not self.train_mode:
                del item_data["similar_question_seq"]
                del item_data["similar_question_diff_seq"]
                del item_data["similar_question_mask_seq"]


class LPKTDataset(BasicSequentialKTDataset):
    def __init__(self, dataset_config, objects):
        super(LPKTDataset, self).__init__(dataset_config, objects)

    def convert_dataset(self):
        id_keys, seq_keys = get_keys_from_kt_data(self.dataset_original)
        self.dataset_converted = {k: [] for k in (id_keys + seq_keys)}
        if "time_seq" in seq_keys:
            self.dataset_converted["interval_time_seq"] = []
        max_seq_len = len(self.dataset_original[0]["mask_seq"])
        for _, item_data in enumerate(self.dataset_original):
            seq_len = item_data["seq_len"]
            for k in id_keys:
                self.dataset_converted[k].append(item_data[k])
            for k in seq_keys:
                if k == "time_seq":
                    interval_time_seq = [0]
                    for time_i in range(1, seq_len):
                        # 原始数据以s为单位
                        interval_time_real = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                        interval_time_idx = max(0, min(interval_time_real, 60 * 24 * 30))
                        interval_time_seq.append(interval_time_idx)
                    interval_time_seq += [0] * (max_seq_len - seq_len)
                    self.dataset_converted["interval_time_seq"].append(interval_time_seq)
                elif k == "use_time_seq":
                    use_time_seq = list(map(lambda t: max(0, min(t, 60 * 60)), item_data["use_time_seq"]))
                    self.dataset_converted[k].append(use_time_seq)
                else:
                    self.dataset_converted[k].append(item_data[k])

        if "time_seq" in self.dataset_converted.keys():
            del self.dataset_converted["time_seq"]


class LBKTDataset(BasicSequentialKTDataset):
    def __init__(self, dataset_config, objects):
        super(LBKTDataset, self).__init__(dataset_config, objects)

    def load_dataset(self):
        setting_name = self.dataset_config["setting_name"]
        file_name = self.dataset_config["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), "LBKT", file_name)
        self.dataset_original = read_kt_file(dataset_path)

    def dataset2tensor(self):
        self.dataset = {}
        for k in self.dataset_converted.keys():
            if k not in ["hint_factor_seq", "attempt_factor_seq", "time_factor_seq"]:
                self.dataset[k] = torch.tensor(self.dataset_converted[k]).long().to(self.dataset_config["device"])
            else:
                self.dataset[k] = torch.tensor(self.dataset_converted[k]).float().to(self.dataset_config["device"])
