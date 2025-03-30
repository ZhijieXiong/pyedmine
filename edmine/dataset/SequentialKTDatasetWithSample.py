import numpy as np
import torch
from copy import deepcopy

from edmine.dataset.SequentialKTDataset import BasicSequentialKTDataset
from edmine.dataset.Sampler import *


class CL4KTDataset(BasicSequentialKTDataset):
    def __init__(self, dataset_config, objects, train_mode=False):
        self.train_mode = train_mode
        self.data_sampler = None
        super(CL4KTDataset, self).__init__(dataset_config, objects)

    def __len__(self):
        return len(self.dataset_original)

    def __getitem__(self, index):
        result = dict()
        
        for key in self.dataset.keys():
            result[key] = self.dataset[key][index]
            
        max_seq_len = result["mask_seq"].shape[0]

        dataset_config_this = self.params["datasets_config"][self.params["datasets_config"]["dataset_this"]]
        aug_type = dataset_config_this["kt4aug"]["aug_type"]
        item_data2aug = deepcopy(self.data_uniformed[index])
        
        if "age_seq" in item_data2aug.keys():
            del item_data2aug["age_seq"]
            
        random_select_aug_len = dataset_config_this["kt4aug"][aug_type].get("random_select_aug_len", False)
        seq_len = item_data2aug["seq_len"]
        if random_select_aug_len and seq_len > 10:
            seq_len = random.randint(10, seq_len)
        for k, v in item_data2aug.items():
            if type(v) == list:
                item_data2aug[k] = v[:seq_len]
                if random_select_aug_len and k not in ["time_seq", "use_time_seq", "interval_time_seq"]:
                    result[f"{k}_random_len"] = torch.tensor(
                        item_data2aug[k] + [0] * (max_seq_len - seq_len)
                    ).long().to(self.params["device"])
        item_data2aug["seq_len"] = seq_len
        # 使用hard neg
        use_hard_neg = dataset_config_this["kt4aug"][aug_type].get("use_hard_neg", False)
        hard_neg_prob = dataset_config_this["kt4aug"][aug_type].get("hard_neg_prob", 1)
        if use_hard_neg:
            correct_seq_neg = CL4KTSampker.negative_seq(item_data2aug["correct_seq"], hard_neg_prob)
            result["correct_seq_hard_neg"] = (
                torch.tensor(correct_seq_neg + [0] * (max_seq_len - seq_len)).long().to(self.params["device"]))
            
        datas_aug = self.get_random_aug(item_data2aug)
        
        # 补零
        for i, data_aug in enumerate(datas_aug):
            pad_len = max_seq_len - data_aug["seq_len"]
            for k, v in data_aug.items():
                if type(v) == list and k not in ["time_seq", "use_time_seq", "interval_time_seq", "age_seq"]:
                    # 数据增强不考虑时间、年龄
                    result[f"{k}_aug_{i}"] = torch.tensor(v + [0] * pad_len).long().to(self.params["device"])
                    
        for key, value in result.items():
            if key not in ["hint_factor_seq", "attempt_factor_seq", "time_factor_seq", "answer_score_seq"]:
                result[key] = torch.tensor(value).long().to(self.dataset_config["device"])
            else:
                result[key] = torch.tensor(value).float().to(self.dataset_config["device"])

        return result

    def process_dataset(self):
        self.load_dataset()
        self.data_sampler = CL4KTSampker(self.dataset_original)
        
    def get_random_aug(self, item_data2aug):
        num_aug = self.datasets_config["num_aug"]
        random_aug_config = self.datasets_config["random_aug"]
        aug_order = random_aug_config["aug_order"]
        mask_prob = random_aug_config["mask_prob"]
        replace_prob = random_aug_config["replace_prob"]
        permute_prob = random_aug_config["permute_prob"]
        crop_prob = random_aug_config["crop_prob"]
        aug_result = []
        for _ in range(num_aug):
            item_data_aug = deepcopy(item_data2aug)
            for aug_type in aug_order:
                if aug_type == "mask":
                    item_data_aug = CL4KTSampker.mask_seq(item_data_aug, mask_prob, 10)
                elif aug_type == "replace":
                    item_data_aug = self.data_sampler.replace_seq(item_data_aug, replace_prob)
                elif aug_type == "permute":
                    item_data_aug = CL4KTSampker.permute_seq(item_data_aug, permute_prob, 10)
                elif aug_type == "crop":
                    item_data_aug = CL4KTSampker.crop_seq(item_data_aug, crop_prob, 10)
                else:
                    raise NotImplementedError()
            item_data_aug["seq_len"] = len(item_data_aug["mask_seq"])
            aug_result.append(item_data_aug)
        return aug_result