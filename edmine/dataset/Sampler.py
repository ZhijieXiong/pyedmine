import random
import math
import numpy as np
from collections import defaultdict


class CLKTSampker:
    def __init__(self, data_uniformed):
        self.easier_concepts = None
        self.harder_concepts = None
        self.get_concept_difficulty(data_uniformed)

    def get_concept_difficulty(self, data_uniformed):
        concept_seqs = [item_data["concept_seq"][:item_data["seq_len"]] for item_data in data_uniformed]
        correctness_seqs = [item_data["correctness_seq"][:item_data["seq_len"]] for item_data in data_uniformed]

        concept_correctness = defaultdict(int)
        concept_count = defaultdict(int)
        for concept_seq, correctness_seq in zip(concept_seqs, correctness_seqs):
            for c_id, correctness in zip(concept_seq, correctness_seq):
                concept_correctness[c_id] += correctness
                concept_count[c_id] += 1

        concept_difficulty = {
            c_id: concept_correctness[c_id] / float(concept_count[c_id]) for c_id in concept_correctness
        }
        concept_ordered_difficulty = [
            item[0] for item in sorted(concept_difficulty.items(), key=lambda x: x[1])
        ]
        easier_concepts = {}
        harder_concepts = {}
        for i, s in enumerate(concept_ordered_difficulty):
            if i == 0:
                # the hardest
                easier_concepts[s] = concept_ordered_difficulty[i + 1]
                harder_concepts[s] = s
            elif i == len(concept_ordered_difficulty) - 1:
                # the easiest
                easier_concepts[s] = s
                harder_concepts[s] = concept_ordered_difficulty[i - 1]
            else:
                easier_concepts[s] = concept_ordered_difficulty[i + 1]
                harder_concepts[s] = concept_ordered_difficulty[i - 1]
        self.easier_concepts = easier_concepts
        self.harder_concepts = harder_concepts

    def replace_seq(self, sample, replace_prob):
        seq_len = sample["seq_len"]
        replace_idx = random.sample(list(range(seq_len)), k=max(1, int(seq_len * replace_prob)))
        for i in replace_idx:
            c_id = sample["concept_seq"][i]
            correctness = sample["correctness_seq"][i]
            if correctness == 0 and c_id in self.harder_concepts.keys():
                # if the response is wrong, then replace a skill with the harder one
                similar_c = self.harder_concepts[c_id]
            elif correctness == 1 and c_id in self.easier_concepts.keys():
                # if the response is correct, then replace a skill with the easier one
                similar_c = self.easier_concepts[c_id]
            else:
                similar_c = sample["concept_seq"][i]

            sample["concept_seq"][i] = similar_c

        return sample

    @staticmethod
    def mask_seq(sample, mask_prob, mask_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < mask_min_seq_len:
            return

        seq_keys = []
        for k in sample.keys():
            if type(sample[k]) == list:
                seq_keys.append(k)

        mask_idx = random.sample(list(range(seq_len)), k=max(1, int(seq_len * mask_prob)))
        for i in mask_idx:
            for k in seq_keys:
                sample[k][i] = -1
        for k in seq_keys:
            sample[k] = list(filter(lambda x: x != -1, sample[k]))
        sample["seq_len"] = len(sample["correctness_seq"])

    @staticmethod
    def permute_seq(sample, perm_prob, perm_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < perm_min_seq_len:
            return

        seq_keys = []
        for k in sample.keys():
            if type(sample[k]) == list:
                seq_keys.append(k)
        reorder_seq_len = max(2, math.floor(perm_prob * seq_len))
        # count和not_permute用于控制while True的循环次数，当循环次数超过一定次数，都没能得到合适的start_pos时，跳出循环，不做置换
        count = 0
        not_permute = False
        while True:
            if count >= 50:
                not_permute = True
                break
            count += 1
            start_pos = random.randint(0, seq_len - reorder_seq_len)
            if start_pos + reorder_seq_len < seq_len:
                break
        if not_permute:
            return

        perm = np.random.permutation(reorder_seq_len)
        for k in seq_keys:
            seq = sample[k]
            sample[k] = seq[:start_pos] + np.asarray(seq[start_pos:start_pos + reorder_seq_len])[perm]. \
                tolist() + seq[start_pos + reorder_seq_len:]

    @staticmethod
    def crop_seq(sample, crop_prob, crop_min_seq_len=10):
        seq_len = sample["seq_len"]
        if seq_len < crop_min_seq_len:
            return

        seq_keys = []
        for k in sample.keys():
            if type(sample[k]) == list:
                seq_keys.append(k)
        cropped_seq_len = min(seq_len - 1, math.floor((1 - crop_prob) * seq_len))
        count = 0
        not_crop = False
        while True:
            if count >= 50:
                not_crop = True
                break
            count += 1
            start_pos = random.randint(0, seq_len - cropped_seq_len)
            if start_pos + cropped_seq_len < seq_len:
                break
        if not_crop:
            return sample

        for k in seq_keys:
            sample[k] = sample[k][start_pos: start_pos + cropped_seq_len]
        sample["seq_len"] = len(sample["correctness_seq"])

    @staticmethod
    def negative_seq(correctness_seq, neg_prob):
        seq_len = len(correctness_seq)
        negative_idx = random.sample(list(range(seq_len)), k=int(seq_len * neg_prob))
        for i in negative_idx:
            correctness_seq[i] = 1 - correctness_seq[i]

        return correctness_seq
