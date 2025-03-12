import numpy as np
import torch
import os

from torch.utils.data import Dataset


def read_triple(file_path, entity2id, relation2id, original=False):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            if original:
                triples.append((h, r, t))
            else:
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def read_id_map(file_path):
    id_map = dict()
    with open(file_path, "r") as fin:
        for line in fin:
            id_, ele = line.strip().split('\t')
            id_map[ele] = int(id_)
    return id_map


class KG4EXDataset(Dataset):
    def __init__(self, dataset_config, objects):
        super(KG4EXDataset, self).__init__()
        self.dataset_config = dataset_config
        self.objects = objects
        self.triples = None
        self.triple_set = None
        self.load_dataset()

        self.count = self.count_frequency()
        self.true_head, self.true_tail = self.get_true_head_and_tail()

    def load_dataset(self):
        setting_name = self.dataset_config["setting_name"]
        file_name = self.dataset_config["file_name"]
        dataset_path = os.path.join(self.objects["file_manager"].get_setting_dir(setting_name), "kg4ex", file_name)
        self.triples = read_triple(dataset_path, self.objects["dataset"]["entity2id"], self.objects["dataset"]["relation2id"])
        self.triple_set = set(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        is_train = self.dataset_config["is_train"]
        negative_sample_size = self.dataset_config.get("negative_sample_size", None)
        mode = self.dataset_config["mode"]
        num_entity = len(self.objects["dataset"]["entity2id"])

        if is_train:
            positive_sample = self.triples[idx]
            head, relation, tail = positive_sample
            subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight])).to(self.dataset_config["device"])

            negative_sample_list = []
            num_negative_sample = 0
            while num_negative_sample < negative_sample_size:
                negative_sample = np.random.randint(num_entity, size=negative_sample_size * 2)
                if mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True
                    )
                elif mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError(f'Training batch mode {mode} not supported')
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                num_negative_sample += negative_sample.size

            negative_sample = np.concatenate(negative_sample_list)[:negative_sample_size]
            negative_sample = torch.LongTensor(negative_sample).to(self.dataset_config["device"])
            positive_sample = torch.LongTensor(positive_sample).to(self.dataset_config["device"])
            return positive_sample, negative_sample, subsampling_weight, mode
        else:
            head, relation, tail = self.triples[idx]
            if mode == 'head-batch':
                tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                       else (-1, head) for rand_head in range(num_entity)]
                tmp[head] = (0, head)
            elif mode == 'tail-batch':
                tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                       else (-1, tail) for rand_tail in range(num_entity)]
                tmp[tail] = (0, tail)
            else:
                raise ValueError(f'negative batch mode {mode} not supported')

            tmp = torch.LongTensor(tmp).to(self.dataset_config["device"])
            filter_bias = tmp[:, 0].float()
            negative_sample = tmp[:, 1]
            positive_sample = torch.LongTensor((head, relation, tail)).to(self.dataset_config["device"])
            return positive_sample, negative_sample, filter_bias, mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    def count_frequency(self, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for head, relation, tail in self.triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    def get_true_head_and_tail(self):
        true_head = {}
        true_tail = {}

        for head, relation, tail in self.triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data
