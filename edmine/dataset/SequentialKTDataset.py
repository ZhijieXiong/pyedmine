import os
import torch

from torch.utils.data import Dataset

from edmine.utils.data_io import read_kt_file
from edmine.utils.parse import get_keys_from_kt_data


class BasicSequentialKTDataset(Dataset):
    def __init__(self, dataset_config, objects):
        super(BasicSequentialKTDataset, self).__init__()
        self.dataset_config = dataset_config
        self.objects = objects
        self.dataset = None

        self.load_dataset()

    def __len__(self):
        return len(self.dataset["mask_seq"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
        return result

    def load_dataset(self):
        setting_name = self.dataset_config["setting_name"]
        file_name = self.dataset_config["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        dataset_original = read_kt_file(dataset_path)
        id_keys, seq_keys = get_keys_from_kt_data(dataset_original)

        dataset_converted = {k: [] for k in (id_keys + seq_keys)}
        if "time_seq" in seq_keys:
            dataset_converted["interval_time_seq"] = []
        max_seq_len = len(dataset_original[0]["mask_seq"])
        for seq_i, item_data in enumerate(dataset_original):
            seq_len = item_data["seq_len"]
            for k in id_keys:
                dataset_converted[k].append(item_data[k])
            for k in seq_keys:
                if k == "time_seq":
                    interval_time_seq = [0]
                    for time_i in range(1, seq_len):
                        interval_time_real = (item_data["time_seq"][time_i] - item_data["time_seq"][time_i - 1]) // 60
                        interval_time_idx = max(0, interval_time_real)
                        interval_time_seq.append(interval_time_idx)
                    interval_time_seq += [0] * (max_seq_len - seq_len)
                    dataset_converted["interval_time_seq"].append(interval_time_seq)
                else:
                    dataset_converted[k].append(item_data[k])

        if "time_seq" in dataset_converted.keys():
            del dataset_converted["time_seq"]

        for k in dataset_converted.keys():
            if k not in ["hint_factor_seq", "attempt_factor_seq", "time_factor_seq", "correct_float"]:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.dataset_config["device"])
            else:
                dataset_converted[k] = torch.tensor(dataset_converted[k]).float().to(self.dataset_config["device"])
        self.dataset = dataset_converted

    def get_statics_kt_dataset(self):
        num_seq = len(self.dataset["mask_seq"])
        with torch.no_grad():
            num_sample = torch.sum(self.dataset["mask_seq"][:, 1:]).item()
            num_interaction = torch.sum(self.dataset["mask_seq"]).item()
            correctness_seq = self.dataset["correctness_seq"]
            mask_bool_seq = torch.ne(self.dataset["mask_seq"], 0)
            num_correct = torch.sum(torch.masked_select(correctness_seq, mask_bool_seq)).item()
        return num_seq, num_sample, num_correct / num_interaction
