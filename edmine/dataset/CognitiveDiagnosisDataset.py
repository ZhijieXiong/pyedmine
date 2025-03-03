import os
import torch

from torch.utils.data import Dataset

from edmine.utils.data_io import read_cd_file


class BasicCognitiveDiagnosisDataset(Dataset):
    def __init__(self, dataset_config, objects):
        super(BasicCognitiveDiagnosisDataset, self).__init__()
        self.dataset_config = dataset_config
        self.objects = objects
        self.dataset = None
        self.load_dataset()

    def __len__(self):
        return len(self.dataset["user_id"])

    def __getitem__(self, index):
        result = dict()
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
        return result

    def load_dataset(self):
        setting_name = self.dataset_config["setting_name"]
        file_name = self.dataset_config["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), file_name)
        dataset_original = read_cd_file(dataset_path)
        all_keys = list(dataset_original[0].keys())
        dataset_converted = {k: [] for k in all_keys}
        for interaction_data in dataset_original:
            for k in all_keys:
                dataset_converted[k].append(interaction_data[k])

        for k in dataset_converted.keys():
            dataset_converted[k] = torch.tensor(dataset_converted[k]).long().to(self.dataset_config["device"])
        self.dataset = dataset_converted
