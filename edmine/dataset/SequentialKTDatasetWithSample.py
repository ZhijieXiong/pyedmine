import torch
from copy import deepcopy

from edmine.dataset.SequentialKTDataset import BasicSequentialKTDataset
from edmine.dataset.Sampler import *


class CLKTDataset(BasicSequentialKTDataset):
    def __init__(self, dataset_config, objects, train_mode=False):
        self.data_sampler = None
        self.train_mode = train_mode
        super(CLKTDataset, self).__init__(dataset_config, objects)

    def __len__(self):
        return len(self.dataset_original)

    def __getitem__(self, index):
        if self.train_mode:
            return self.getitem_train_mode(index)
        else:
            return self.getitem_test_mode(index)
    
    def getitem_test_mode(self, index):
        result = dict()
        for k, v in self.dataset_original[index].items():
            if k in ["concept_seq", "correctness_seq", "mask_seq"] or type(v) is not list:
                result[k] = torch.tensor(v).long().to(self.dataset_config["device"])
            
        return result
    
    def getitem_train_mode(self, index):
        result = dict()
        user_data2aug = {}
        for k, v in self.dataset_original[index].items():
            if k in ["concept_seq", "correctness_seq", "mask_seq"] or type(v) is not list:
                result[k] = self.dataset_original[index][k]
                user_data2aug[k] = deepcopy(self.dataset_original[index][k])
            
        max_seq_len = len(result["mask_seq"])
        seq_len = user_data2aug["seq_len"]
        for k, v in user_data2aug.items():
            result[k] = torch.tensor(v).long().to(self.dataset_config["device"])
            if type(v) == list:
                user_data2aug[k] = v[:seq_len]
        hard_neg_prob = self.dataset_config.get("hard_neg_prob", 1)
        correctness_seq_neg = CLKTSampker.negative_seq(user_data2aug["correctness_seq"], hard_neg_prob)
        result["correctness_seq_hard_neg"] = (
            torch.tensor(correctness_seq_neg + [0] * (max_seq_len - seq_len)).long().to(self.dataset_config["device"]))
        datas_aug = self.get_random_aug(user_data2aug)
        
        # 补零
        for i, data_aug in enumerate(datas_aug):
            pad_len = max_seq_len - data_aug["seq_len"]
            for k, v in data_aug.items():
                if type(v) == list:
                    result[f"{k}_aug_{i}"] = torch.tensor(v + [0] * pad_len).long().to(self.dataset_config["device"])
        return result

    def process_dataset(self):
        self.load_dataset()
        self.add_concept_seq()
        self.data_sampler = CLKTSampker(self.dataset_original)
        
    def add_concept_seq(self):
        q2c = self.objects["dataset"]["q2c"]
        for user_data in self.dataset_original:
            user_data["concept_seq"] = list(map(lambda q_id: q2c[q_id][0], user_data["question_seq"]))
        
    def get_random_aug(self, user_data2aug):
        num_aug = self.dataset_config["num_aug"]
        aug_order = self.dataset_config["aug_order"]
        mask_prob = self.dataset_config["mask_prob"]
        replace_prob = self.dataset_config["replace_prob"]
        permute_prob = self.dataset_config["permute_prob"]
        crop_prob = self.dataset_config["crop_prob"]
        aug_result = []
        for _ in range(num_aug):
            user_data_aug = deepcopy(user_data2aug)
            for aug_type in aug_order:
                if aug_type == "mask":
                    CLKTSampker.mask_seq(user_data_aug, mask_prob, 10)
                elif aug_type == "replace":
                    self.data_sampler.replace_seq(user_data_aug, replace_prob)
                elif aug_type == "permute":
                    CLKTSampker.permute_seq(user_data_aug, permute_prob, 10)
                elif aug_type == "crop":
                    CLKTSampker.crop_seq(user_data_aug, crop_prob, 10)
                else:
                    raise NotImplementedError()
            user_data_aug["seq_len"] = len(user_data_aug["mask_seq"])
            aug_result.append(user_data_aug)
        return aug_result
    