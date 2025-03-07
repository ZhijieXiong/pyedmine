import torch
import json
import math
import os
import numpy as np
import torch.nn as nn


class EmbedLayer(nn.Module):
    """
    Manages multiple embedding layers based on configuration.
    Allows control over whether embeddings are learnable during training.
    Supports three types of embedding initialization:
        - Default Initialization: Uses nn.Embedding with random initialization.
        - Custom Initialization: Initializes embeddings with specific methods (e.g., constant initialization for correctness embeddings).
        - Pre-trained Initialization: Loads embeddings from a pre-trained file.
    Input embed_configs: dict[str, dict], key and value of embed_config are
        - num_item: The number of unique items (e.g., number of questions).
        - dim_item: The dimensionality of the embedding.
        - learnable: (Optional) A boolean indicating whether the embedding is learnable (default: True).
        - init_method: (Optional) Specifies the initialization method (e.g., "init_correctness_1", "init_correctness_2").
        - embed_path: (Optional) Path to a pre-trained embedding file. If provided, embeddings are loaded from this file.
    """
    def __init__(self, embed_configs: dict[str, dict]):
        super(EmbedLayer, self).__init__()
        self.embed_configs = embed_configs

        for embed_name, embed_config in embed_configs.items():
            if (("embed_path" not in embed_config) and
                    ("learnable" not in embed_config) and
                    ("init_method" not in embed_config)):
                # 默认nn.Embedding
                self.__setattr__(embed_name, nn.Embedding(embed_config["num_item"], embed_config["dim_item"]))
            elif "embed_path" not in embed_config:
                # 根据init_method使用不同初始化方法
                init_method = embed_config.get("init_method", "default")
                if init_method in ["init_correctness_1", "init_correctness_2"]:
                    # 初始化correctness，默认是不可学习的
                    self.__setattr__(embed_name, self.init_constant_embed(embed_config))
                else:
                    self.__setattr__(embed_name, nn.Embedding(embed_config["num_item"], embed_config["dim_item"]))
                    if init_method == "init_zero":
                        nn.init.constant_(self.__getattr__(embed_name).weight, 0.)
                    if init_method == "xavier_normal":
                        nn.init.xavier_normal_(self.__getattr__(embed_name).weight)
                    # 默认是可学习的
                    self.__getattr__(embed_name).weight.requires_grad = embed_config.get("learnable", True)
            else:
                self.__setattr__(embed_name, self.init_embed_from_pretrained(embed_name, embed_config))

    @staticmethod
    def init_constant_embed(embed_config):
        """
        初始化固定的embed layer，如下\n
        1、init_correctness_1：(2, dim)，0用全0表示，1用全1表示\n
        2、init_correctness_2：(2, dim)，0用左边一半元素为1，右边一半元素为0表示，1则相反

        :param embed_config:
        :return:
        """
        init_method = embed_config["init_method"]
        dim_item = embed_config["dim_item"]
        if init_method == "init_correctness_1":
            embed = nn.Embedding(2, dim_item)
            embed.weight.data[0] = torch.zeros(dim_item)
            embed.weight.data[1] = torch.ones(dim_item)
        elif init_method == "init_correctness_2":
            dim_half = dim_item // 2
            embed = nn.Embedding(2, dim_item)
            embed.weight.data[0, :dim_half] = 0
            embed.weight.data[0, dim_half:] = 1
            embed.weight.data[1, :dim_half] = 1
            embed.weight.data[1, dim_half:] = 0
        else:
            raise NotImplementedError()
        embed.weight.requires_grad = False
        return embed

    def init_embed_from_pretrained(self, embed_name, embed_config):
        """

        :param embed_name:
        :param embed_config:
        :return:
        """
        num_item = embed_config["num_item"]
        dim_item = embed_config["dim_item"]
        embed_path = embed_config["embed_path"]

        if not os.path.exists(embed_path):
            raise ValueError(f"embed_path `{embed_path}` does not exist")

        with open(embed_path, 'r') as f:
            precomputed_embeddings = json.load(f)
        pretrained_emb_tensor = torch.tensor(
            [precomputed_embeddings[str(i)] for i in range(len(precomputed_embeddings))], dtype=torch.float)

        num_emb, dim_emb = pretrained_emb_tensor.shape

        assert num_item == num_emb

        # Normalize the lengths to 1, for convenience.
        norms = pretrained_emb_tensor.norm(p=2, dim=1, keepdim=True)
        pretrained_emb_tensor = pretrained_emb_tensor / norms
        # Now scale to expected size.
        pretrained_emb_tensor = pretrained_emb_tensor * np.sqrt(num_item)

        if dim_item != dim_emb:
            self.__setattr__(f"{embed_name}ProjectionLayer", nn.Linear(dim_emb, dim_item))

        return nn.Embedding.from_pretrained(pretrained_emb_tensor, freeze=not embed_config.get("learnable", True))

    def get_emb(self, embed_name, item_index):
        """
        获取指定embed里的emb

        :param embed_name:
        :param item_index:
        :return:
        """
        embed_config = self.embed_configs[embed_name]

        if "embed_path" not in embed_config:
            return self.__getattr__(embed_name)(item_index)
        else:
            if hasattr(self, f"{embed_name}ProjectionLayer"):
                return self.__getattr__(f"{embed_name}ProjectionLayer")(
                    self.__getattr__(embed_name)(item_index)
                )
            else:
                return self.__getattr__(embed_name)(item_index)

    def get_emb_concatenated(self, cat_order, item_index2cat):
        """
        Concatenates embeddings from multiple embedding layers in a specified order.
        Returns a single tensor representing the combined embeddings for each item in the batch.
        :param cat_order: A list specifying the order in which embeddings should be concatenated. Each element in the list corresponds to the name of an embedding layer (e.g., ["question", "correctness"]).
        :param item_index2cat: A list of tensors, where each tensor contains indices for retrieving embeddings from the corresponding embedding layer.
        :return:
        """
        concatenated_emb = self.get_emb(cat_order[0], item_index2cat[0])
        for i, embed_name in enumerate(cat_order[1:]):
            concatenated_emb = torch.cat((concatenated_emb, self.get_emb(embed_name, item_index2cat[i + 1])), dim=-1)
        return concatenated_emb

    def get_emb_fused1(
            self,
            related_embed_name: str,
            base2related_transfer_table,
            base2related_mask_table,
            base_item_index,
            fusion_method="mean",
            other_item_index=None
    ):
        """
        Fuses embeddings of related items (e.g., concepts) associated with a base item (e.g., questions) using a specified fusion method (e.g., mean pooling).
        :param related_embed_name: The name of the embedding layer for the related items (e.g., "concept").
        :param base2related_transfer_table: base2related_transfer_table: A tensor of shape (num_base_items, max_num_related_items) that maps each base item to its related items. For example, if the base is a question and the related items are concepts, this table stores the concept IDs for each question. Padding is used if a question has fewer concepts than max_num_related_items.
        :param base2related_mask_table: A tensor of shape (num_base_items, max_num_related_items) that acts as a mask for the base2related_transfer_table. It contains 1 for valid related items and 0 for padding.
        :param base_item_index: A tensor of indices specifying the base items for which fused embeddings are to be computed.
        :param fusion_method: The method used to fuse the embeddings.
        :param other_item_index: for DKVMN
        :return:
        """
        embed_related = self.__getattr__(related_embed_name)
        if other_item_index is None:
            related_emb = embed_related(base2related_transfer_table[base_item_index])
        else:
            related_emb = embed_related(base2related_transfer_table[base_item_index] + other_item_index)
        mask = base2related_mask_table[base_item_index]
        if fusion_method == "mean":
            related_emb_fusion = (related_emb * mask.unsqueeze(-1)).sum(-2)
            related_emb_fusion = related_emb_fusion / mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return related_emb_fusion

    def get_emb_fused2(
            self,
            related2_embed_name,
            base2related1_transfer_table,
            base2related1_mask_table,
            related_1_to_2_transfer_table,
            base_item_index,
            fusion_method="mean"
    ):
        """
        Fuses embeddings of second-level related items (e.g., concept difficulties) associated with first-level related items (e.g., concepts), which are linked to a base item (e.g., questions).
        :param related2_embed_name: The name of the embedding layer for the second-level related items (e.g., "concept_difficulty").
        :param base2related1_transfer_table: A tensor of shape (num_base_items, max_num_related1_items) that maps each base item to its first-level related items (e.g., concepts). Padding is used if a base item has fewer related items than max_num_related1_items.
        :param base2related1_mask_table: A tensor of shape (num_base_items, max_num_related1_items) that acts as a mask for base2related1_transfer_table. It contains 1 for valid first-level related items and 0 for padding.
        :param related_1_to_2_transfer_table: A tensor of shape (num_related1_items,) that maps each first-level related item (e.g., concept) to its corresponding second-level related item (e.g., concept difficulty).
        :param base_item_index: A tensor of indices specifying the base items for which fused embeddings are to be computed.
        :param fusion_method: The method used to fuse the embeddings.
        :return:
        """
        embed_related2 = self.__getattr__(related2_embed_name)
        related1_item_index = base2related1_transfer_table[base_item_index]
        related2_emb = embed_related2(related_1_to_2_transfer_table[related1_item_index])
        mask = base2related1_mask_table[base_item_index]
        if fusion_method == "mean":
            related2_emb_fusion = (related2_emb * mask.unsqueeze(-1)).sum(-2)
            related2_emb_fusion = related2_emb_fusion / mask.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return related2_emb_fusion


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(CosinePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]
    