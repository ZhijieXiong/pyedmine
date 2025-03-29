import os
import math
import re
import time
import datetime

import pandas as pd
import numpy as np

from typing import Any
from copy import deepcopy

import edmine.constant.kt_dataset as CONSTANT
from edmine.utils.data_io import read_csv, read_json
from edmine.data.FileManager import FileManager
from edmine.utils.parse import get_kt_data_statics


def get_info_function(df: pd.DataFrame, col_name: str) -> int:
    return len(list(filter(lambda x: str(x) != "nan", pd.unique(df[col_name]))))


def load_ednet_kt1(data_dir, num_file):
    # 多知识点算新知识点
    dfs = []

    def process_tags(tags_str):
        # 多知识点是用_连接的，但是如 1_2_3 和 2_3_1 表示同一多知识点组合，所以统一表示成id小的在前面，即1_2_3
        tags = tags_str.split("_")
        tags = list(map(str, sorted(list(map(int, tags)))))
        return "_".join(tags)

    for i in range(num_file):
        file_name = f"users_{i}.csv"
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
        df["tags"] = df["tags"].map(process_tags)
        dfs.append(df)

    return pd.concat(dfs, axis=0)


def load_SLP(data_dir, dataset_name):
    subject = dataset_name.split("-")[-1]
    unit_path = os.path.join(data_dir, f"unit-{subject}.csv")
    term_path = os.path.join(data_dir, f"term-{subject}.csv")
    student_path = os.path.join(data_dir, "student.csv")
    family_path = os.path.join(data_dir, "family.csv")
    # school_path = os.path.join(data_dir, "school.csv")

    useful_cols = ["student_id", "question_id", "concept", "score", "full_score", "time_access"]
    family_cols = ["student_id", "live_on_campus"]
    student_cols = ["student_id", "gender", "school_id"]
    # school_cols = ["school_id", "school_type"]

    unit = read_csv(unit_path, useful_cols)
    term = read_csv(term_path, useful_cols)
    student = read_csv(student_path, student_cols)
    family = read_csv(family_path, family_cols)
    # school = load_csv(school_path, school_cols)

    # 原文件已经是排过序的，加上order方便后面利用
    unit["order"] = range(len(unit))
    term["order"] = range(len(unit), len(unit) + len(term))
    # 将总评数据加入
    student_ids = pd.unique(unit["student_id"])
    student_df = pd.DataFrame({"student_id": student_ids})

    # unit为0，term为1
    unit.insert(loc=len(unit.columns), column='interaction_type', value=0)
    term = student_df.merge(term, how="left", on=["student_id"])
    term.insert(loc=len(term.columns), column='interaction_type', value=1)
    df = pd.concat([unit, term], axis=0)

    df = df.merge(family, how="left", on=["student_id"])
    df = df.merge(student, how="left", on=["student_id"])
    # df = df.merge(school, how="left", on=["school_id"])

    # live_on_campus和school_type有nan
    return df[["student_id", "question_id", "concept", "score", "full_score", "time_access", "order",
               "live_on_campus", "school_id", "gender", "interaction_type"]]


def map_qc_id(df: pd.DataFrame) -> dict[str, Any]:
    """
    Remaps question IDs and concept IDs in a dataset to sequential integers, constructs a question-concept relationship matrix (Q_table), and returns the processed data along with mapping information for both questions and concepts.
    :param df: A DataFrame containing at least two columns: `question_id` and `concept_id`
    :return: A dict containing the following elements. ||
        `data_processed`: The input DataFrame with remapped question_id and concept_id columns. ||
        `Q_table`: A binary matrix (numpy.ndarray) where rows represent questions, columns represent concepts, and 1 indicates a relationship between a question and a concept. ||
        `concept_id_map`: A DataFrame mapping original concept IDs to their remapped sequential IDs. ||
        `question_id_map`: A DataFrame mapping original question IDs to their remapped sequential IDs. ||
    """
    concept_ids = pd.unique(df["concept_id"])
    question_ids = pd.unique(df["question_id"])
    question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
    concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
    df["question_id"] = df["question_id"].map(question_id_map)
    df["concept_id"] = df["concept_id"].map(concept_id_map)

    df_new = pd.DataFrame({
        "question_id": map(int, df["question_id"].tolist()),
        "concept_id": map(int, df["concept_id"].tolist())
    })
    Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
    for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
        correspond_c = pd.unique(group_info["concept_id"]).tolist()
        Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)

    return {
        "data_processed": df,
        "Q_table": Q_table,
        "concept_id_map": pd.DataFrame({
            "original_id": concept_id_map.keys(),
            "mapped_id": concept_id_map.values()
        }),
        "question_id_map": pd.DataFrame({
            "original_id": question_id_map.keys(),
            "mapped_id": question_id_map.values()
        })
    }


def map_user_info(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """
    Remaps a specified field in a DataFrame to sequential integers, sorted by the number of users associated with each field value. It also returns mapping information and statistics about the remapped field, including the number of users and interactions for each value.
    :param df: A DataFrame containing at least two columns: user_id and the specified field (e.g., school_id).
    :param field_name: The column name in the DataFrame to be remapped (e.g., school_id).
    :return: info_id_map
    """
    num_user_in_field = df[df[field_name] != -1].groupby(field_name).agg(user_count=("user_id", lambda x: x.nunique())).to_dict()
    num_user_in_field = list(num_user_in_field["user_count"].items())
    num_user_in_field = sorted(num_user_in_field, key=lambda item: item[1], reverse=True)
    field_id_map = {item[0]: i for i, item in enumerate(num_user_in_field)}
    field_id_map[-1] = -1
    df[field_name] = df[field_name].map(field_id_map)

    num_user_in_field = list(map(lambda item: (field_id_map[item[0]], item[1]), num_user_in_field))
    num_user_dict = {}
    for field_id_mapped, num_user in num_user_in_field:
        num_user_dict[field_id_mapped] = num_user

    data2map = {
        field_name: [],
        f"{field_name}_mapped": [],
        "num_user": [],
        "num_interaction": []
    }
    for field_id, field_id_mapped in field_id_map.items():
        if field_id_mapped != -1:
            data2map[field_name] .append(field_id_map[field_id])
            data2map[f"{field_name}_mapped"].append(field_id_mapped)
            data2map["num_user"].append(num_user_dict[field_id_mapped])
            data2map["num_interaction"].append(len(df[df[field_name] == field_id_mapped]))
        else:
            data2map[field_name].append(-1)
            data2map[f"{field_name}_mapped"].append(-1)
            data2map["num_user"].append(len(pd.unique(df[df[field_name] == -1]["user_id"])))
            data2map["num_interaction"].append(len(df[df[field_name] == -1]))

    return pd.DataFrame({
        "original_id": list(data2map[field_name]),
        "mapped_id": list(data2map[f"{field_name}_mapped"]),
        "num_user": list(data2map["num_user"]),
        "num_interaction": list(data2map["num_interaction"]),
    })


class KTDataProcessor:
    def __init__(self, process_config, file_manager: FileManager):
        self.params = process_config
        self.file_manager = file_manager

        self.data_raw = None
        self.statics_raw = None
        self.data_preprocessed = None
        self.statics_preprocessed = None
        self.Q_table = None
        self.user_id_map = None
        self.question_id_map = None
        self.concept_id_map = None
        self.data_uniformed = None
        # 一些数据的信息，如学校、城市
        self.other_info = {}

    @staticmethod
    def get_basic_info(df):
        useful_cols = {"question_id", "concept_id", "concept_name"}
        useful_cols = useful_cols.intersection(set(df.columns))
        result = {
            "num_interaction": len(df),
            "num_user": len(pd.unique(df["user_id"]))
        }
        for col in useful_cols:
            if col == "question_id":
                result["num_question"] = get_info_function(df, "question_id")
            elif col == "concept_id":
                result["num_concept"] = get_info_function(df, "concept_id")
            elif col == "concept_name":
                result["num_concept_name"] = get_info_function(df, "concept_name")
        return result

    def preprocess_data(self):
        datasets_treatable = self.file_manager.builtin_datasets
        dataset_name = self.params["dataset_name"]
        assert dataset_name in datasets_treatable, f"DataProcessor can't handle {dataset_name}"

        data_path = self.params["data_path"]
        assert os.path.exists(data_path), f"raw data ({data_path}) not exist"

        if dataset_name == "assist2009":
            self.process_assist2009()
        elif dataset_name == "assist2009-full":
            self.process_assist2009_full()
        elif dataset_name == "assist2012":
            self.process_assist2012()
        elif dataset_name == "assist2015":
            self.process_assist2015()
        elif dataset_name == "assist2017":
            self.process_assist2017()
        elif dataset_name == "poj":
            self.process_poj()
        elif dataset_name == "edi2020-task1":
            self.process_edi2020_task1()
        elif dataset_name == "edi2020-task34":
            self.process_edi2020_task34()
        elif dataset_name == "ednet-kt1":
            self.process_ednet_kt1()
        elif dataset_name == "xes3g5m":
            self.process_xes3g5m()
        # elif dataset_name in ["algebra2005", "algebra2006", "algebra2008", "bridge2algebra2006", "bridge2algebra2008"]:
        #     self.process_kdd_cup2010()
        elif dataset_name in ["SLP-bio", "SLP-chi", "SLP-eng", "SLP-geo", "SLP-his", "SLP-mat", "SLP-phy"]:
            self.process_SLP()
        elif dataset_name == "statics2011":
            self.process_statics2011()
        elif dataset_name == "slepemapy-anatomy":
            self.process_slepemapy_anatomy()
        elif dataset_name == "junyi2015":
            self.process_junyi2015()
        else:
            raise NotImplementedError()

        if dataset_name == "assist2009":
            self.uniform_assist2009()
        elif dataset_name in ["assist2012", "assist2017", "slepemapy-anatomy", "junyi2015"]:
            self.uniform_assist2012()
        elif dataset_name == "assist2015":
            self.uniform_assist2015()
        elif dataset_name == "poj":
            self.uniform_poj()
        elif dataset_name in ["edi2020-task1", "edi2020-task34"]:
            self.uniform_edi2020()
        elif dataset_name == "xes3g5m":
            # 直接在process_xes3g5m里一起处理了
            pass
        elif dataset_name in ["assist2009-full", "ednet-kt1", "algebra2005", "algebra2006", "algebra2008",
                              "bridge2algebra2006", "bridge2algebra2008", "slepemapy-anatomy"]:
            self.uniform_raw_is_single_concept()
        elif dataset_name in ["SLP-bio", "SLP-chi", "SLP-eng", "SLP-geo", "SLP-his", "SLP-mat", "SLP-phy"]:
            self.uniform_SLP()
        elif dataset_name == "statics2011":
            self.uniform_statics2011()
        else:
            raise NotImplementedError()

        data_statics = get_kt_data_statics(self.data_uniformed, self.Q_table)
        if self.statics_preprocessed is None:
            self.statics_preprocessed = {
                "num_user": data_statics["num_seq"],
                "num_interaction": data_statics["num_sample"],
                "num_concept": self.Q_table.shape[1],
                "num_question": self.Q_table.shape[0],
            }
        self.statics_preprocessed["average_seq_len"] = data_statics["ave_seq_len"]
        self.statics_preprocessed["average_question_acc"] = data_statics["ave_que_acc"]
        self.statics_preprocessed["question_sparsity"] = data_statics["que_sparsity"]
        self.statics_preprocessed["concept_sparsity"] = data_statics["concept_sparsity"]

        return self.data_uniformed

    def process_assist2009(self):
        data_path = self.params["data_path"]
        dataset_name = "assist2009"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 有知识点名称的interaction数量为325637，总数量为401756
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)
        # 该数据集中use_time_first_attempt, num_hint, num_attempt都没有nan，有4684条数据use_time_first_attempt <= 0
        # 关于num attempt和num hint，有脏数据，如attempt的数量大于10，或者为0（官网对attempt count的定义是Number of student attempts on this problem，
        # 没说是从第一次做开始就计数，还是做错了一次后开始计数，我们按照LBKT的设定，假设可以为0）
        # num attempt和num hint都无nan，且都>=0
        df["use_time_first_attempt"] = df["use_time_first_attempt"].map(lambda t: max(1, math.ceil(t / 1000)))
        df["use_time"] = df["use_time"].map(lambda t: max(1, math.ceil(t / 1000)))

        # 获取concept name和concept 原始id的对应并保存
        concept_names = list(pd.unique(df.dropna(subset=["concept_name"])["concept_name"]))
        concept_id2name = {}
        for c_name in concept_names:
            concept_data = df[df["concept_name"] == c_name]
            c_id = int(concept_data["concept_id"].iloc[0])
            concept_id2name[c_id] = c_name.strip()

        process_result = map_qc_id(df)
        self.data_preprocessed = process_result["data_processed"]
        self.Q_table = process_result["Q_table"]
        self.statics_preprocessed = self.get_basic_info(process_result["data_processed"])
        self.question_id_map = process_result["question_id_map"]
        self.concept_id_map = process_result["concept_id_map"]
        self.concept_id_map["text"] = self.concept_id_map["original_id"].map(lambda x: concept_id2name.get(x, ""))

    def uniform_assist2009(self):
        df = deepcopy(self.data_preprocessed)
        # school_id按照学生数量重映射
        df["school_id"] = df["school_id"].fillna(-1)
        df["school_id"] = df["school_id"].map(int)
        self.other_info["school_id_map"] = map_user_info(df, "school_id")

        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "use_time_seq": "use_time",
            "use_time_first_seq": "use_time_first_attempt",
            "num_hint_seq": "num_hint",
            "num_attempt_seq": "num_attempt"
        }
        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"order_id", "concept_name", "concept_id"})
        seq_keys = CONSTANT.datasets_seq_keys()["assist2009"]

        # 去除多知识点习题的冗余
        df = df[~df.duplicated(subset=["user_id", "order_id", "question_id"])]
        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["order_id"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    object_data[seq_key].append(row_data[info_name_table[seq_key]])
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)
        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_assist2009_full(self):
        data_path = self.params["data_path"]
        dataset_name = "assist2009-full"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        df = df[~df['concept_name'].isnull()]

        # 只有8条数据use_time_first_attempt为nan，没有数据num_attempt为nan或者小于0，没有数据answer_type为nan
        df.dropna(subset=["user_id", "question_id", "concept_id", "use_time_first_attempt"], inplace=True)
        df["use_time_first_attempt"] = df["use_time_first_attempt"].map(lambda t: max(1, math.ceil(t / 1000)))
        df["correctness"] = df["correctness"].map(int)

        df["school_id"] = df["school_id"].fillna(-1)
        df["school_id"] = df["school_id"].map(int)
        self.other_info["school_id_map"] = map_user_info(df, "school_id")

        df["class_id"] = df["class_id"].fillna(-1)
        df["class_id"] = df["class_id"].map(int)
        self.other_info["class_id_map"] = map_user_info(df, "class_id")

        q2c_id_map = pd.DataFrame({
            "question_id": list(map(int, df["question_id"].tolist())),
            "concept_name": df["concept_name"].tolist(),
            "concept_id": df["concept_id"].tolist(),
            "question_type": df["question_type"].tolist(),
        })
        concept_id2name = {}
        question_id2type = {}
        for q_id, group_info in q2c_id_map.groupby("question_id"):
            c_ids = pd.unique(group_info["concept_id"]).tolist()[0].split(";")
            c_names = pd.unique(group_info["concept_name"]).tolist()[0].split(";")
            q_type = pd.unique(group_info["question_type"]).tolist()[0]
            for c_id, c_name in zip(c_ids, c_names):
                concept_id2name[int(c_id)] = c_name
            question_id2type[q_id] = q_type.strip()

        self.process_raw_is_single_concept(df)

        self.question_id_map["type"] = self.question_id_map["original_id"].map(lambda x: question_id2type.get(x, ""))
        self.concept_id_map["text"] = self.concept_id_map["original_id"].map(lambda x: concept_id2name.get(x, ""))

    def process_assist2012(self):
        def time_str2timestamp(time_str):
            if len(time_str) != 19:
                time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        data_path = self.params["data_path"]
        dataset_name = "assist2012"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = KTDataProcessor.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 有知识点名称的interaction数量为2630080，总数量为2711813
        df.dropna(subset=["question_id", "concept_id"], inplace=True)

        # 获取concept name和concept 原始id的对应并保存
        concept_names = list(pd.unique(df.dropna(subset=["concept_name"])["concept_name"]))
        concept_id2name = {}
        for c_name in concept_names:
            concept_data = df[df["concept_name"] == c_name]
            c_id = int(concept_data["concept_id"].iloc[0])
            concept_id2name[c_id] = c_name.strip()

        df["correctness"] = df["correctness"].astype('int8')
        # 该数据集中use_time_first_attempt, num_hint, num_attempt都没有nan，有1条数据use_time_first_attempt <= 0
        df["use_time"] = df["use_time"].map(lambda t: max(1, math.ceil(t / 1000)))
        df["use_time_first_attempt"] = df["use_time_first_attempt"].map(
            lambda t: max(1, math.ceil(t / 1000))
        )
        df["timestamp"] = df["timestamp"].map(time_str2timestamp)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)

        process_result = map_qc_id(df)
        self.data_preprocessed = process_result["data_processed"]
        self.Q_table = process_result["Q_table"]
        self.statics_preprocessed = self.get_basic_info(process_result["data_processed"])
        self.question_id_map = process_result["question_id_map"]
        self.concept_id_map = process_result["concept_id_map"]
        self.concept_id_map["text"] = self.concept_id_map["original_id"].map(lambda x: concept_id2name.get(x, ""))

    def process_assist2015(self):
        data_path = self.params["data_path"]
        dataset_name = "assist2015"
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, rename_dict=rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 常规处理，先丢弃correct不为0或1的数据（可能为小数）
        # df = df[(df["correct"] == 0) | (df["correct"] == 1)]
        # 保留为小数分数的记录，大于0.5，correct为1
        df["answer_score"] = deepcopy(df["correctness"])
        df["correctness"] = df["answer_score"].map(lambda s: 0 if s <= 0.5 else 1)
        df["correctness"] = df["correctness"].map(int)
        df["question_id"] = df["question_id"].map(int)
        question_ids = pd.unique(df["question_id"])
        df["question_id"] = df["question_id"].map({q_id: i for i, q_id in enumerate(question_ids)})
        self.data_preprocessed = df
        self.statics_preprocessed = KTDataProcessor.get_basic_info(df)
        self.question_id_map = pd.DataFrame({
            "original_id": question_ids,
            "mapped_id": range(len(question_ids))
        })
        self.concept_id_map = self.question_id_map
        self.Q_table = np.eye(len(self.question_id_map), dtype=int)

    def uniform_assist2015(self):
        df = deepcopy(self.data_preprocessed)
        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "answer_score_seq": "answer_score"
        }

        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"log_id"})
        seq_keys = CONSTANT.datasets_seq_keys()["assist2015"]

        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["log_id"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    if seq_key not in ["answer_score_seq"]:
                        object_data[seq_key].append(int(row_data[info_name_table[seq_key]]))
                    else:
                        object_data[seq_key].append(row_data[info_name_table[seq_key]])
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)
        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_assist2017(self):
        data_path = self.params["data_path"]
        dataset_name = "assist2017"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        # skill字段有noskill值，过滤掉
        # num_attempt和num_hint无NaN值
        df = deepcopy(self.data_raw[self.data_raw["concept_id"] != "noskill"])
        df["use_time"] = df["use_time"].map(lambda t: max(1, int(t)))

        concept_name2id = {}
        concept_id2name = {}
        concept_names = pd.unique(df["concept_id"])
        for i, concept_name in enumerate(concept_names):
            concept_name2id[concept_name.strip()] = i
            concept_id2name[i] = concept_name.strip()
        df["concept_id"] = df.apply(lambda x: concept_name2id[x["concept_id"]], axis=1)

        # 里面有些习题id对应多个知识点id（但实际不是同一个习题），对这些习题id进行重映射，使之成为单知识点数据集
        question_concept_pairs = {}
        for i in df.index:
            question_concept_pair = str(int(df["question_id"][i])) + "," + str(int(df["concept_id"][i]))
            question_concept_pairs.setdefault(question_concept_pair, len(question_concept_pairs))
            df["question_id"][i] = question_concept_pairs[question_concept_pair]

        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        df["question_id"] = df["question_id"].map(int)
        df["concept_id"] = df["concept_id"].map(int)

        process_result = map_qc_id(df)
        self.data_preprocessed = process_result["data_processed"]
        self.Q_table = process_result["Q_table"]
        self.statics_preprocessed = self.get_basic_info(process_result["data_processed"])
        # assist2017的习题id经过了两次映射
        qc_pairs_reverse = {v: int(k.split(",")[0]) for k, v in question_concept_pairs.items()}
        self.question_id_map = process_result["question_id_map"]
        self.question_id_map["original_id"] = self.question_id_map["original_id"].map(qc_pairs_reverse)
        self.concept_id_map = process_result["concept_id_map"]
        self.concept_id_map["text"] = self.concept_id_map["original_id"].map(lambda x: concept_id2name.get(x, ""))

    def uniform_assist2012(self):
        df = deepcopy(self.data_preprocessed)
        dataset_name = self.params["dataset_name"]

        if dataset_name in ["assist2012", "assist2017"]:
            # school_id按照学生数量重映射
            df["school_id"] = df["school_id"].fillna(-1)
            df["school_id"] = df["school_id"].map(int)
            self.other_info["school_id_map"] = map_user_info(df, "school_id")
        if dataset_name in ["slepemapy-anatomy"]:
            df["country_id"] = df["country_id"].fillna(-1)
            self.other_info["country_id_map"] = map_user_info(df, "country_id")

        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "time_seq": "timestamp",
            "use_time_seq": "use_time",
            "use_time_first_seq": "use_time_first_attempt",
            "num_hint_seq": "num_hint",
            "num_attempt_seq": "num_attempt"
        }

        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"concept_name", "concept_id"})
        seq_keys = CONSTANT.datasets_seq_keys()[dataset_name]

        user_id_map = {}
        seqs = []
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    # 这几个数据集的seq都是int类型
                    object_data[seq_key].append(int(row_data[info_name_table[seq_key]]))
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)
        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_statics2011(self):
        data_path = self.params["data_path"]
        dataset_name = "statics2011"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # num_hint无nan，无<0的值
        df["num_hint"] = df["num_hint"].fillna(0)

        def replace_text(text):
            return text.strip("_")

        def time_str2timestamp(time_str):
            datetime_obj = datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M")
            timestamp = int(datetime_obj.timestamp())
            return timestamp

        def process_concept_id(c_id_str):
            # 格式为[ccc, xxx, cid]的层级知识点，取最细粒度的作为concept
            c_ids = c_id_str.split(",")
            c_id = c_ids[-1]
            c_id = c_id.strip()
            return c_id

        df.dropna(subset=['Problem Name', 'Step Name', 'timestamp', 'correctness'], inplace=True)
        df["concept_id"] = df["concept_id"].apply(process_concept_id)
        df["timestamp"] = df["timestamp"].apply(time_str2timestamp)
        df = df[df["correctness"] != "hint"]
        df.loc[df["correctness"] == "correct", "correctness"] = 1
        df.loc[df["correctness"] == "incorrect", "correctness"] = 0

        df['Problem Name'] = df['Problem Name'].apply(replace_text)
        df['Step Name'] = df['Step Name'].apply(replace_text)
        df["question_id"] = df.apply(lambda x: f"question: {x['Problem Name']}, step: {x['Step Name']}", axis=1)
        df["question_id"] = df["question_id"].map(str)

        question_names = pd.unique(df["question_id"])
        question_name2id = {}
        question_id2name = {}
        for i, question_name in enumerate(question_names):
            question_name2id[question_name.strip()] = i
            question_id2name[i] = question_name.strip()
        df["question_id"] = df.apply(lambda x: question_name2id[x["question_id"]], axis=1)

        concept_names = pd.unique(df["concept_id"])
        concept_name2id = {}
        concept_id2name = {}
        for i, concept_name in enumerate(concept_names):
            concept_name2id[concept_name.strip()] = i
            concept_id2name[i] = concept_name.strip()
        df["concept_id"] = df.apply(lambda x: concept_name2id[x["concept_id"]], axis=1)

        df["user_id"] = df["user_id"].map(str)
        df = df[["user_id", "timestamp", "concept_id", "correctness", "question_id", "num_hint"]]

        process_result = map_qc_id(df)

        self.data_preprocessed = process_result["data_processed"]
        self.Q_table = process_result["Q_table"]
        self.statics_preprocessed = self.get_basic_info(process_result["data_processed"])
        self.question_id_map = process_result["question_id_map"]
        self.question_id_map["text"] = self.question_id_map["original_id"].map(lambda x: question_id2name.get(x, ""))
        self.concept_id_map = process_result["concept_id_map"]
        self.concept_id_map["text"] = self.concept_id_map["original_id"].map(lambda x: concept_id2name.get(x, ""))

    def uniform_statics2011(self):
        df = deepcopy(self.data_preprocessed)
        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "time_seq": "timestamp",
            "num_hint_seq": "num_hint"
        }

        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"concept_name", "concept_id"})
        seq_keys = CONSTANT.datasets_seq_keys()["statics2011"]

        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    object_data[seq_key].append(row_data[info_name_table[seq_key]])
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)
        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_poj(self):
        def time_str2timestamp(time_str):
            if len(time_str) != 19:
                time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        data_path = self.params["data_path"]
        dataset_name = "poj"
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, rename_dict=rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        # 该数据集是编程习题，只有Accepted表示做对，其它或多或少都有错误；且数据集中没有nan
        df["error_type"] = deepcopy(df["correctness"])
        df["correctness"] = df["correctness"].map(lambda s: 1 if s == "Accepted" else 0)
        df["correctness"] = df["correctness"].map(int)
        df["question_id"] = df["question_id"].map(int)
        question_ids = pd.unique(df["question_id"])
        df["question_id"] = df["question_id"].map({q_id: i for i, q_id in enumerate(question_ids)})
        error_types = pd.unique(df["error_type"])
        df["error_type"] = df["error_type"].map({type_id: i for i, type_id in enumerate(error_types)})
        df["timestamp"] = df["timestamp"].map(time_str2timestamp)

        self.data_preprocessed = df
        self.statics_preprocessed = KTDataProcessor.get_basic_info(df)
        self.question_id_map = pd.DataFrame({
            "original_id": question_ids,
            "mapped_id": range(len(question_ids))
        })
        self.concept_id_map = self.question_id_map
        self.other_info["error_type_id_map"] = pd.DataFrame({
            "original_id": error_types,
            "mapped_id": range(len(error_types))
        })
        self.Q_table = np.eye(len(self.question_id_map), dtype=int)

    def uniform_poj(self):
        df = deepcopy(self.data_preprocessed)
        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "time_seq": "timestamp",
            "error_type_seq": "error_type"
        }
        id_keys = list(set(df.columns) - set(info_name_table.values()))
        seq_keys = CONSTANT.datasets_seq_keys()["poj"]

        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    object_data[seq_key].append(row_data[info_name_table[seq_key]])
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)
        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_junyi2015(self):
        dataset_name = "junyi2015"
        data_dir = self.params["data_path"]
        data_path = os.path.join(data_dir, "junyi_ProblemLog_original.csv")
        metadata_question_path = os.path.join(data_dir, "junyi_Exercise_table.csv")
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]

        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        # todo: 结合area列和prerequisites列构造层次知识点
        meta_question_cols = ["name", "topic"]
        meta_question_rename_map = {
            "name": "question_name",
            "topic": "concept_name",
        }
        metadata_question = read_csv(metadata_question_path, meta_question_cols, meta_question_rename_map)
        metadata_question.dropna(subset=["question_name", "concept_name"], inplace=True)

        metadata_question["question_id"] = range(len(metadata_question))
        concept_names = pd.unique(metadata_question["concept_name"])
        concept_name2id = {c_name: i for i, c_name in enumerate(concept_names)}
        metadata_question["concept_id"] = metadata_question["concept_name"].map(concept_name2id)

        self.question_id_map = deepcopy(metadata_question[["question_name", "question_id"]])
        self.question_id_map.rename(columns={"question_name": "text", "question_id": "mapped_id"}, inplace=True)
        self.question_id_map["original_id"] = self.question_id_map["mapped_id"]
        self.concept_id_map = pd.DataFrame({
            "concept_id": concept_name2id.values(),
            "concept_name": concept_name2id.keys()
        })
        self.concept_id_map.rename(columns={"concept_name": "text", "concept_id": "mapped_id"}, inplace=True)
        self.concept_id_map["original_id"] = self.concept_id_map["mapped_id"]
        num_question = len(self.question_id_map)
        num_concept = len(self.concept_id_map)
        Q_table = np.zeros((num_question, num_concept), dtype=int)
        for row in metadata_question.iterrows():
            q_id = row[1]["question_id"]
            c_id = row[1]["concept_id"]
            Q_table[q_id, c_id] = 1
        self.Q_table = Q_table

        df = deepcopy(self.data_raw)
        df.dropna(subset=["question_name"], inplace=True)
        df = df.merge(metadata_question, how="left")
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        # 有nan的列：use_time_first
        df["use_time_first_attempt"] = df["use_time_first_attempt"].fillna(0)
        df["use_time_first_attempt"] = df["use_time_first_attempt"].map(
            lambda time_str: max(0, math.ceil(list(map(int, str(time_str).split("&")))[0] / 1000))
        )
        df["use_time"] = df["use_time"].map(lambda t: 1 if (t <= 0) else t)
        df["timestamp"] = df["timestamp"].map(lambda x: int(x / 1000000))
        df["correctness"] = df["correctness"].map(int)
        self.data_preprocessed = df[["user_id", "question_id", "concept_id", "correctness", "timestamp", "use_time",
                                     "use_time_first_attempt", "num_hint", "num_attempt"]]
        self.statics_preprocessed = self.get_basic_info(self.data_preprocessed)
        # 这个数据集的习题和知识点数量不是统计处理的，而是直接从junyi_Exercise_table这里面获取的
        self.statics_preprocessed["num_concept"] = num_concept
        self.statics_preprocessed["num_question"] = num_question

    def process_ednet_kt1(self):
        data_dir = self.params["data_path"]
        # data_dir下每个文件存放了5000名学生（随机但每个文件下学生不重复）的记录，num_file指定要读取几个文件
        # 这里随机选5000名学生
        self.data_raw = load_ednet_kt1(data_dir, num_file=1)
        dataset_name = "ednet-kt1"
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw.rename(columns=rename_cols, inplace=True)

        df = deepcopy(self.data_raw)
        df["use_time"] = df["use_time"].map(lambda t: max(1, int(t) // 1000))
        df["timestamp"] = df["timestamp"].map(lambda t: t // 1000)
        self.process_raw_is_single_concept(df)

    def process_slepemapy_anatomy(self):
        data_path = self.params["data_path"]
        dataset_name = "slepemapy-anatomy"
        useful_cols = CONSTANT.datasets_useful_cols()[dataset_name]
        rename_cols = CONSTANT.datasets_renamed()[dataset_name]
        self.data_raw = read_csv(data_path, useful_cols, rename_cols)
        self.statics_raw = self.get_basic_info(self.data_raw)

        def time_str2timestamp(time_str):
            if len(time_str) != 19:
                time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        def process_concept(context_name):
            c_id = context_name.strip("'").strip("\"").strip()
            return c_id

        def process_question(one_row):
            q_type_ = one_row["question_type"].strip("'").strip("\"").strip()
            c_name = one_row["concept_id"].strip("'").strip("\"").strip()
            item_asked = one_row["item_asked"]
            q_id_ = f"{q_type_}@@{c_name}@@{item_asked}"

            return q_id_

        # 这个数据集来自一个医学习题网站（分辨人体器官组织，选择题），里面有两种习题：(t2d) find the given term on the image; (d2t) pick the name for the highlighted term
        # 用得上的字段: item_asked、item_answered（empty if the user answered "I don't know"）分别表示问的术语和回答的术语，context_name：name of the image (context)
        # item_asked、context_name和type组成一道question的要素
        # 该习题库是层级的知识点systems_asked -> locations_asked (9个location，以这个作为知识点，太笼统) -> item_asked
        # 本框架下选择context_name作为知识点
        # todo: location_asked可以作为更高层级的知识点
        df = deepcopy(self.data_raw)
        df.dropna(subset=["user_id", "timestamp", "question_type", "concept_id", "item_asked", "use_time"], inplace=True)
        df["concept_id"] = df["concept_id"].apply(process_concept)
        df["question_id"] = df.apply(process_question, axis=1)
        df["timestamp"] = df["timestamp"].map(time_str2timestamp)
        df["use_time"] = df["use_time"].map(lambda t: min(max(1, int(t) // 1000), 60 * 60))
        df["correctness"] = df["item_asked"] == df["item_answered"]
        df["correctness"] = df["correctness"].map(int)

        question_id_map = {}
        question_id2type = {}
        for i, (q_id, group_info) in enumerate(df.groupby("question_id")):
            q_type = pd.unique(group_info["question_type"]).tolist()[0]
            question_id2type[q_id] = q_type.strip()
            question_id_map[q_id] = i
        df["question_id"] = df["question_id"].map(question_id_map)
        question_id_map1 = pd.DataFrame({
            "merge_id": question_id_map.keys(),
            "original_id": question_id_map.values(),
            "mapped_id": question_id_map.values(),
        })
        question_id_map2 = pd.DataFrame({
            "merge_id": question_id2type.keys(),
            "type": question_id2type.values()
        })
        self.question_id_map = pd.merge(question_id_map1, question_id_map2, on='merge_id', how='left')
        del self.question_id_map["merge_id"]

        concept_ids = pd.unique(df["concept_id"])
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        df["concept_id"] = df["concept_id"].map(concept_id_map)
        self.concept_id_map = pd.DataFrame({
            "original_id": concept_id_map.values(),
            "mapped_id": concept_id_map.values(),
            "text": concept_id_map.keys()
        })

        self.data_preprocessed = df[
            ["user_id", "question_id", "correctness", "timestamp", "use_time", "country_id", "concept_id"]
        ]
        self.statics_preprocessed = KTDataProcessor.get_basic_info(self.data_preprocessed)

        # Q table
        df_new = pd.DataFrame({
            "question_id": map(int, df["question_id"].tolist()),
            "concept_id": map(int, df["concept_id"].tolist())
        })
        Q_table = np.zeros((len(question_id_map), len(concept_ids)), dtype=int)
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table = Q_table

    def process_raw_is_single_concept(self, df):
        """
        处理像ednet-kt1、assist2009-full和kdd-cup这种数据，其每条记录是一道习题\n
        这类数据由分为两类，一类是concept id为数字（ednet-kt1、assist2009-full）；
        todo: 另一类是concept id非数值（algebra2005、algebra2006、bridge2algebra2006、algebra2008、bridge2algebra2008）
        """
        dataset_name = self.params["dataset_name"]
        # 习题id重映射
        question_ids = list(pd.unique(df["question_id"]))
        df["question_id"] = df["question_id"].map(
            {q_id: i for i, q_id in enumerate(question_ids)})
        self.question_id_map = pd.DataFrame({
            "original_id": question_ids,
            "mapped_id": range(len(question_ids))
        })

        # 知识点id重映射
        split_table = {
            "ednet-kt1": "_",
            "assist2009-full": ";",
            "algebra2005": "~~",
            "algebra2006": "~~",
            "algebra2008": "~~",
            "bridge2algebra2006": "~~",
            "bridge2algebra2008": "~~",
            "slepemapy-anatomy": "@@"
        }
        df_ = df[~df.duplicated(subset=["question_id"])]
        c_ids_strs = list(pd.unique(df_["concept_id"]))
        concept_ids = set()
        for c_ids_str in c_ids_strs:
            c_ids = c_ids_str.split(split_table[dataset_name])
            for c_id in c_ids:
                concept_ids.add(c_id)
        c_id_map = {}
        Q_table = np.zeros((len(df_), len(concept_ids)), dtype=int)
        for q_id, c_ids_str in zip(df_["question_id"].tolist(), df_["concept_id"].tolist()):
            c_ids = c_ids_str.split(split_table[dataset_name])
            c_ids_mapped = []
            for c_id in c_ids:
                if c_id not in c_id_map:
                    c_id_map[c_id] = len(c_id_map)
                c_ids_mapped.append(c_id_map[c_id])
            Q_table[[q_id] * len(c_ids_mapped), c_ids_mapped] = [1] * len(c_ids_mapped)
        del df["concept_id"]
        self.data_preprocessed = df
        self.concept_id_map = pd.DataFrame({
            "original_id": c_id_map.keys(),
            "mapped_id": c_id_map.values()
        })
        self.Q_table = Q_table

    def uniform_raw_is_single_concept(self):
        dataset_name = self.params["dataset_name"]
        df = deepcopy(self.data_preprocessed)
        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "time_seq": "timestamp",
            "use_time_seq": "use_time",
            "num_hint_seq": "num_hint",
            "num_attempt_seq": "num_attempt",
            "use_time_first_seq": "use_time_first_attempt",
        }
        order_key_table = {
            "ednet-kt1": ["timestamp"],
            "assist2009-full": ["order_id"],
            "algebra2005": ["timestamp"],
            "algebra2006": ["timestamp"],
            "algebra2008": ["timestamp"],
            "bridge2algebra2006": ["timestamp"],
            "bridge2algebra2008": ["timestamp"],
            "slepemapy-anatomy": ["timestamp"]
        }
        if dataset_name in ["algebra2005", "algebra2006", "algebra2008", "bridge2algebra2006", "bridge2algebra2008"]:
            df["tmp_index"] = range(df.shape[0])
            order_key_table[dataset_name].append("tmp_index")

        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"concept_id", "concept_name", "question_type"} - set(order_key_table[dataset_name]))
        seq_keys = CONSTANT.datasets_seq_keys()[dataset_name]

        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=order_key_table[dataset_name])
            if dataset_name == "ednet-kt1":
                user_data["timestamp"] = user_data["timestamp"].map(lambda x: int(x / 1000))
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            if "tmp_index" in object_data.keys():
                del object_data["tmp_index"]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    object_data[seq_key].append(int(row_data[info_name_table[seq_key]]))
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)

        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_edi2020_task1(self):
        # todo: 提取层次结构知识点
        data_dir = self.params["data_path"]
        data_train_path = os.path.join(data_dir, "train_data", "train_task_1_2.csv")
        task1_test_public_path = os.path.join(data_dir, "test_data", "test_public_answers_task_1.csv")
        task1_test_private_path = os.path.join(data_dir, "test_data", "test_private_answers_task_1.csv")
        metadata_answer_path = os.path.join(data_dir, "metadata", "answer_metadata_task_1_2.csv")
        metadata_question_path = os.path.join(data_dir, "metadata", "question_metadata_task_1_2.csv")
        metadata_student_path = os.path.join(data_dir, "metadata", "student_metadata_task_1_2.csv")

        metadata_answer_cols = ["AnswerId", "DateAnswered"]
        df_cols = ["QuestionId", "AnswerId", "UserId", "IsCorrect"]
        df_rename_map = {
            "QuestionId": "question_id",
            "AnswerId": "answer_id",
            "UserId": "user_id",
            "IsCorrect": "correctness"
        }
        meta_question_rename_map = {
            "QuestionId": "question_id",
            "SubjectId": "concept_ids"
        }
        meta_student_raname_map = {
            "UserId": "user_id",
            "Gender": "gender",
            "DateOfBirth": "birth",
        }
        meta_answer_rename_map = {
            "AnswerId": "answer_id",
            "DateAnswered": "timestamp"
        }

        # df_train、df_test_public、df_test_private均无NaN
        # metadata_student在DateOfBirth和PremiumPupil列有NaN
        # metadata_question无NaN
        # metadata_answer在AnswerId、Confidence和SchemeOfWorkId列有NaN
        df_train = read_csv(data_train_path, df_cols, df_rename_map)
        df_task1_test_public = read_csv(task1_test_public_path, df_cols, df_rename_map)
        df_task1_test_private = read_csv(task1_test_private_path, df_cols, df_rename_map)
        metadata_answer = read_csv(metadata_answer_path, metadata_answer_cols, meta_answer_rename_map)
        metadata_question = read_csv(metadata_question_path, rename_dict=meta_question_rename_map)
        metadata_student = read_csv(metadata_student_path, rename_dict=meta_student_raname_map)

        # 0,1,2分别表示train、test pub、test pri
        df_train.insert(loc=len(df_train.columns), column='dataset_type', value=0)
        df_task1_test_public.insert(loc=len(df_task1_test_public.columns), column='dataset_type', value=1)
        df_task1_test_private.insert(loc=len(df_task1_test_private.columns), column='dataset_type', value=2)
        self.data_raw = pd.concat((
            df_train,
            df_task1_test_public,
            df_task1_test_private
        ), axis=0)
        self.statics_raw = KTDataProcessor.get_basic_info(self.data_raw)

        df = deepcopy(self.data_raw)
        metadata_answer.dropna(subset=["answer_id"], inplace=True)
        metadata_answer["answer_id"] = metadata_answer["answer_id"].map(int)

        def time_str2year(time_str):
            if str(time_str) == "nan":
                # 后面用time_year - birth，如果为0或者为负数，说明至少其中一项为NaN
                return 3000
            return int(time_str[:4])

        def time_str2timestamp(time_str):
            # if len(time_str) != 19:
            #     time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        def map_gender(item):
            if item not in [1, 2]:
                return 0
            return item

        time_year = metadata_answer["timestamp"].map(time_str2year)
        time_year.name = "time_year"
        metadata_answer["timestamp"] = metadata_answer["timestamp"].map(time_str2timestamp)
        metadata_answer = pd.concat((metadata_answer, time_year), axis=1)
        metadata_student["birth"] = metadata_student["birth"].map(time_str2year)
        df = df.merge(metadata_student, how="left")
        df["gender"] = df["gender"].map(map_gender)

        # 丢弃没有时间戳的数据
        metadata_answer = metadata_answer[~metadata_answer.duplicated(subset=["answer_id"])]
        df = df.merge(metadata_answer, how="left")
        df.dropna(subset=["timestamp"], inplace=True)

        # 连接metadata_question
        df = df.merge(metadata_question, how="left")
        df.dropna(subset=["concept_ids"], inplace=True)

        # 习题id重映射
        question_ids = list(pd.unique(df["question_id"]))
        question_ids.sort()
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        self.question_id_map = pd.DataFrame({
            "original_id": question_ids,
            "mapped_id": range(len(question_ids))
        })

        # 计算年龄：time_year - birth
        age = df["time_year"] - df["birth"]
        age.name = "age"
        age[age <= 5] = 0
        df = pd.concat([df, age], axis=1)

        # 知识点id重映射，metadata_question中Subject_Id是层级知识点，取最后一个（最细粒度的）知识点作为习题知识点，所以是单知识点数据集
        concept_ids = set()
        question_concept_map = {}
        for i in range(len(metadata_question)):
            q_id = metadata_question.iloc[i]["question_id"]
            c_ids_str = metadata_question.iloc[i]["concept_ids"]
            c_ids = eval(c_ids_str)
            question_concept_map[question_id_map[q_id]] = [c_ids[-1]]
            concept_ids.add(c_ids[-1])
        concept_ids = list(concept_ids)
        concept_ids.sort()
        self.concept_id_map = pd.DataFrame({
            "original_id": concept_ids,
            "mapped_id": range(len(concept_ids))
        })

        # 习题到知识点的映射
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        for q_id in question_concept_map.keys():
            question_concept_map[q_id] = list(map(lambda c_id: concept_id_map[c_id], question_concept_map[q_id]))
        df_question_concept = pd.DataFrame({
            "question_id": question_concept_map.keys(),
            "concept_id": map(lambda c_ids_: c_ids_[0], question_concept_map.values())
        })
        df = df.merge(df_question_concept, how="left", on=["question_id"])
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        self.data_preprocessed = df[["user_id", "question_id", "concept_id", "correctness", "age", "timestamp", "gender",
                                     "dataset_type"]
        ]
        self.statics_preprocessed = KTDataProcessor.get_basic_info(self.data_preprocessed)

        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for q_id in question_concept_map.keys():
            correspond_c = question_concept_map[q_id]
            Q_table[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table = Q_table

    def process_edi2020_task34(self):
        # todo: 提取层次结构知识点
        data_dir = self.params["data_path"]
        data_train_path = os.path.join(data_dir, "train_data", "train_task_3_4.csv")
        metadata_answer_path = os.path.join(data_dir, "metadata", "answer_metadata_task_3_4.csv")
        metadata_question_path = os.path.join(data_dir, "metadata", "question_metadata_task_3_4.csv")
        metadata_student_path = os.path.join(data_dir, "metadata", "student_metadata_task_3_4.csv")

        # df_train、df_test_public、df_test_private均无NaN
        # metadata_student在DateOfBirth和PremiumPupil列有NaN
        # metadata_question无NaN
        # metadata_answer在AnswerId、Confidence和SchemeOfWorkId列有NaN
        df_rename_map = {
            "QuestionId": "question_id",
            "AnswerId": "answer_id",
            "UserId": "user_id",
            "IsCorrect": "correctness"
        }
        meta_question_rename_map = {
            "QuestionId": "question_id",
            "SubjectId": "concept_ids"
        }
        meta_student_raname_map = {
            "UserId": "user_id",
            "Gender": "gender",
            "DateOfBirth": "birth",
        }
        meta_answer_rename_map = {
            "AnswerId": "answer_id",
            "DateAnswered": "timestamp"
        }

        df = read_csv(data_train_path, useful_cols=["QuestionId", "AnswerId", "UserId", "IsCorrect"],
                      rename_dict=df_rename_map)
        metadata_answer = read_csv(metadata_answer_path, useful_cols=["AnswerId", "DateAnswered"],
                                   rename_dict=meta_answer_rename_map)
        metadata_question = read_csv(metadata_question_path, rename_dict=meta_question_rename_map)
        metadata_student = read_csv(metadata_student_path, rename_dict=meta_student_raname_map)

        # 对这个数据集，只使用比赛数据的训练集，在pykt-toolkit框架中就是这么处理的
        metadata_answer.dropna(subset=["answer_id"], inplace=True)
        metadata_answer["answer_id"] = metadata_answer["answer_id"].map(int)

        def time_str2year(time_str):
            if str(time_str) == "nan":
                # 后面用time_year - birth，如果为0或者为负数，说明至少其中一项为NaN
                return 3000
            return int(time_str[:4])

        def time_str2timestamp(time_str):
            # if len(time_str) != 19:
            #     time_str = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", time_str).group()
            return int(time.mktime(time.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S")))

        def map_gender(item):
            # if item not in [1, 2]:
            #     return 0
            if item == 1:
                return 0
            elif item == 2:
                return 1
            else:
                return -1

        time_year = metadata_answer["timestamp"].map(time_str2year)
        time_year.name = "time_year"
        metadata_answer["timestamp"] = metadata_answer["timestamp"].map(time_str2timestamp)
        metadata_answer = pd.concat((metadata_answer, time_year), axis=1)
        metadata_student["birth"] = metadata_student["birth"].map(time_str2year)
        df = df.merge(metadata_student, how="left")
        df["gender"] = df["gender"].map(map_gender)

        # 丢弃没有时间戳的数据
        metadata_answer = metadata_answer[~metadata_answer.duplicated(subset=["answer_id"])]
        df = df.merge(metadata_answer, how="left")
        df.dropna(subset=["timestamp"], inplace=True)

        # 连接metadata_question
        df = df.merge(metadata_question, how="left")
        df.dropna(subset=["concept_ids"], inplace=True)

        # 习题id重映射
        question_ids = list(pd.unique(df["question_id"]))
        question_ids.sort()
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        self.question_id_map = pd.DataFrame({
            "original_id": question_ids,
            "mapped_id": range(len(question_ids))
        })

        # 计算年龄：time_year - birth
        age = df["time_year"] - df["birth"]
        age.name = "age"
        age[age <= 5] = 5
        df = pd.concat([df, age], axis=1)

        # 知识点id重映射，metadata_question中Subject_Id是层级知识点，取最后一个（最细粒度的）知识点作为习题知识点，所以是单知识点数据集
        concept_ids = set()
        question_concept_map = {}
        for i in range(len(metadata_question)):
            q_id = metadata_question.iloc[i]["question_id"]
            c_ids_str = metadata_question.iloc[i]["concept_ids"]
            c_ids = eval(c_ids_str)
            question_concept_map[question_id_map[q_id]] = [c_ids[-1]]
            concept_ids.add(c_ids[-1])
        concept_ids = list(concept_ids)
        concept_ids.sort()
        self.concept_id_map = pd.DataFrame({
            "original_id": concept_ids,
            "mapped_id": range(len(concept_ids))
        })

        # 习题到知识点的映射
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        for q_id in question_concept_map.keys():
            question_concept_map[q_id] = list(map(lambda c_id: concept_id_map[c_id], question_concept_map[q_id]))
        df_question_concept = pd.DataFrame({
            "question_id": question_concept_map.keys(),
            "concept_id": map(lambda c_ids_: c_ids_[0], question_concept_map.values())
        })
        df = df.merge(df_question_concept, how="left", on=["question_id"])
        df.dropna(subset=["question_id", "concept_id"], inplace=True)
        self.data_preprocessed = df[["user_id", "question_id", "concept_id", "correctness", "age", "timestamp", "gender"]]
        self.statics_preprocessed = KTDataProcessor.get_basic_info(self.data_preprocessed)

        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for q_id in question_concept_map.keys():
            correspond_c = question_concept_map[q_id]
            Q_table[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table = Q_table

    def uniform_edi2020(self):
        df = deepcopy(self.data_preprocessed)
        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "time_seq": "timestamp",
            "age_seq": "age"
        }
        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"concept_id"})
        dataset_name = self.params["dataset_name"]
        seq_keys = deepcopy(CONSTANT.datasets_seq_keys()[dataset_name])
        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    object_data[seq_key].append(row_data[info_name_table[seq_key]])
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)

        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_xes3g5m(self):
        # 习题类型：单选和填空
        data_dir = self.params["data_path"]
        # kc_level和question_level的数据是一样的，前者是multi_concept，后者是only_question（对于多知识点习题用 _ 拼接知识点）
        train_valid_path = os.path.join(data_dir, "question_level", "train_valid_sequences_quelevel.csv")
        test_path = os.path.join(data_dir, "question_level", "test_quelevel.csv")
        df_train_valid = read_csv(train_valid_path)[
            ["uid", "questions", "concepts", "responses", "timestamps", "selectmasks"]
        ]
        df_test = read_csv(test_path)[["uid", "questions", "concepts", "responses", "timestamps"]]

        # metadata
        question_meta_path = os.path.join(data_dir, "metadata", "questions.json")
        concept_meta_path = os.path.join(data_dir, "metadata", "kc_routes_map.json")
        question_meta = read_json(question_meta_path)
        concept_meta = read_json(concept_meta_path)
        qs_id, qs_text, qs_answer, qs_analysis, qs_type = [], [], [], [], []
        for q_id, q_meta in question_meta.items():
            qs_id.append(int(q_id))
            qs_text.append(q_meta["content"].strip())
            qs_answer.append(q_meta["answer"])
            qs_analysis.append(q_meta["analysis"].strip())
            qs_type.append(q_meta["type"].strip())
        cs_id, cs_text = [], []
        for c_id, c_text in concept_meta.items():
            c_id = int(c_id)
            if c_id > 864:
                # todo: 提取层级知识点关系
                # id > 864都不是最低层级知识点
                break
            cs_id.append(c_id)
            cs_text.append(c_text.strip())
        self.question_id_map = pd.DataFrame({
            "original_id": qs_id,
            "mapped_id": qs_id,
            "text": qs_text,
            "type": qs_type,
            "answer": qs_answer,
            "analysis": qs_analysis
        })
        self.concept_id_map = pd.DataFrame({
            "original_id": cs_id,
            "mapped_id": cs_id,
            "text": cs_text
        })

        data_all = {}
        for i in df_train_valid.index:
            user_id = int(df_train_valid["uid"][i])
            data_all.setdefault(user_id, {
                "question_seq": [],
                "concept_seq": [],
                "correctness_seq": [],
                "time_seq": []
            })
            # df_train_valid提供的数据是切割好的（将长序列切成固定长度为200的序列），不足200的用-1补齐
            mask_seq = list(map(int, df_train_valid["selectmasks"][i].split(",")))
            if -1 in mask_seq:
                end_pos = mask_seq.index(-1)
            else:
                end_pos = 200

            question_seq = list(map(int, df_train_valid["questions"][i].split(",")))[:end_pos]
            concept_seq = list(map(lambda cs_str: list(map(int, cs_str.split("_"))),
                                   df_train_valid["concepts"][i].split(",")))[:end_pos]
            correctness_seq = list(map(int, df_train_valid["responses"][i].split(",")))[:end_pos]
            time_seq = list(map(int, df_train_valid["timestamps"][i].split(",")))[:end_pos]
            data_all[user_id]["question_seq"] += question_seq
            data_all[user_id]["concept_seq"] += concept_seq
            data_all[user_id]["correctness_seq"] += correctness_seq
            data_all[user_id]["time_seq"] += time_seq

        for i in df_test.index:
            # df_test提供的数据是未切割的
            user_id = int(df_test["uid"][i])
            data_all.setdefault(user_id, {
                "question_seq": [],
                "concept_seq": [],
                "correctness_seq": [],
                "time_seq": []
            })
            question_seq = list(map(int, df_test["questions"][i].split(",")))
            concept_seq = list(map(lambda cs_str: list(map(int, cs_str.split("_"))),
                                   df_test["concepts"][i].split(",")))
            correctness_seq = list(map(int, df_test["responses"][i].split(",")))
            time_seq = list(map(int, df_test["timestamps"][i].split(",")))
            data_all[user_id]["question_seq"] += question_seq
            data_all[user_id]["concept_seq"] += concept_seq
            data_all[user_id]["correctness_seq"] += correctness_seq
            data_all[user_id]["time_seq"] += time_seq

        # 处理成统一格式，即[{user_id(int), question_seq(list), concept_seq(list), correctness_seq(list), time_seq(list)}, ...]
        data_uniformed = [{
            "user_id": user_id,
            "question_seq": seqs["question_seq"],
            "concept_seq": seqs["concept_seq"],
            "correctness_seq": seqs["correctness_seq"],
            "time_seq": list(map(lambda t: int(t/1000), seqs["time_seq"])),
            "seq_len": len(seqs["correctness_seq"])
        } for user_id, seqs in data_all.items()]

        # 提取每道习题对应的知识点：提供的数据（train_valid_sequences_quelevel.csv和test_quelevel.csv）中习题对应的知识点是最细粒度的，类似edi2020数据集中层级知识点里最细粒度的知识点
        # 其中concept_meta里面的知识点也是最细粒度的
        # 而question metadata里每道题的kc routes是完整的知识点（层级）
        # 并且提供的数据中习题对应知识点和question metadata中习题对应的知识点不是完全一一对应的，例如习题1035
        # 在question metadata中对应的知识点为
        # ['拓展思维----应用题模块----年龄问题----年龄问题基本关系----年龄差', '能力----运算求解',
        #  '课内题型----综合与实践----应用题----倍数问题----已知两量之间倍数关系和两量之差，求两个量',
        #  '学习能力----七大能力----运算求解',
        #  '拓展思维----应用题模块----年龄问题----年龄问题基本关系----年龄问题基本关系和差问题',
        #  '课内知识点----数与运算----数的运算的实际应用（应用题）----整数的简单实际问题----除法的实际应用',
        #  '知识点----应用题----和差倍应用题----已知两量之间倍数关系和两量之差，求两个量',
        #  '知识点----数的运算----估算与简单应用----整数的简单实际问题----除法的实际应用']
        # 在数据中对应的知识点为[169, 177, 239, 200, 73]，其对应的知识点名称为['除法的实际应用', '已知两量之间倍数关系和两量之差，求两个量', '年龄差', '年龄问题基本关系和差问题', '运算求解']
        q2c_map = {}
        for item_data in data_uniformed:
            for i in range(item_data["seq_len"]):
                q_id = item_data["question_seq"][i]
                c_ids = item_data["concept_seq"][i]
                q2c_map.setdefault(q_id, c_ids)

        # 习题和知识点id都是映射过的，但是习题共有7651个，其id却是从0开始，7651结束
        # 原因是习题6232在数据集中没出现过，但是在question_meta中是有这道题的，所以保留该习题id
        # 习题6232对应知识点795（减法横式）
        Q_table = np.zeros((len(self.question_id_map), len(self.concept_id_map)), dtype=int)
        for q_id in q2c_map.keys():
            correspond_c = q2c_map[q_id]
            Q_table[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        Q_table[6232, 795] = 1
        self.Q_table = Q_table

        self.data_uniformed = []
        user_id_map = {}
        for i, item_data in enumerate(data_uniformed):
            item_data_ = {}
            user_id_map[item_data["user_id"]] = i
            for k in item_data:
                if k == "user_id":
                    item_data_["user_id"] = i
                    continue
                if k != "concept_seq":
                    item_data_[k] = deepcopy(item_data[k])
            self.data_uniformed.append(item_data_)
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def process_SLP(self):
        data_dir = self.params["data_path"]
        dataset_name = self.params["dataset_name"]
        self.data_raw = load_SLP(data_dir, dataset_name)
        self.data_raw.rename(columns=CONSTANT.datasets_renamed()["SLP"], inplace=True)
        self.statics_raw = self.get_basic_info(self.data_raw)

        # 去除question_id和concept_id为nan以及"n.a."的数据
        df = deepcopy(self.data_raw)
        df.dropna(subset=["question_id", "concept_id", "score", "full_score", "timestamp"], inplace=True)
        df = df[(df["question_id"] != "n.a.") & (df["concept_id"] != "n.a.")]

        # 将user_id、question_id、concept_id、school_type、live_on_campus、timestamp映射为数字
        user_ids = list(pd.unique(df["user_id"]))
        df["user_id"] = df["user_id"].map({u_id: i for i, u_id in enumerate(user_ids)})

        question_ids = list(pd.unique(df["question_id"]))
        concept_ids = list(pd.unique(df["concept_id"]))
        question_id_map = {q_id: i for i, q_id in enumerate(question_ids)}
        concept_id_map = {c_id: i for i, c_id in enumerate(concept_ids)}
        df["question_id"] = df["question_id"].map(question_id_map)
        df["concept_id"] = df["concept_id"].map(concept_id_map)
        question_info = pd.DataFrame({
            "original_id": question_id_map.keys(),
            "mapped_id": question_id_map.values()
        })
        concept_info = pd.DataFrame({
            "original_id": concept_id_map.keys(),
            "mapped_id": concept_id_map.values()
        })
        self.question_id_map = question_info
        self.concept_id_map = concept_info

        def map_campus(item):
            item_str = str(item)
            if item_str == "Yes":
                return 1
            if item_str == "No":
                return 2
            return 0

        df["live_on_campus"] = df["live_on_campus"].map(map_campus)

        def map_gender(item):
            item_str = str(item)
            if item_str == "Male":
                return 2
            if item_str == "Female":
                return 1
            return 0

        df["gender"] = df["gender"].map(map_gender)

        def time_str2timestamp(time_str):
            if "-" in time_str:
                return int(time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S")))
            else:
                return int(time.mktime(time.strptime(time_str, "%Y/%m/%d %H:%M")))

        df["timestamp"] = df["timestamp"].map(time_str2timestamp)
        df["order"] = df["order"].map(int)

        # 将score和full_score转换为correct
        df["score"] = df["score"].map(float)

        def map_full_score(item):
            item_str = str(item)
            if item_str == "n.a.":
                return 1.0
            return float(item_str)

        df["full_score"] = df["full_score"].map(map_full_score)
        df["answer_score"] = df["score"] / df["full_score"]
        df["correctness"] = df["answer_score"] > 0.5
        df["correctness"] = df["correctness"].map(int)
        df["answer_score"] = df["answer_score"].map(lambda s: float("{:.2f}".format(s)))

        df.rename(columns={"live_on_campus": "campus"}, inplace=True)
        self.data_preprocessed = df[
            ["user_id", "question_id", "concept_id", "correctness", "gender", "campus", "school_id", "timestamp", "order",
             "interaction_type", "answer_score"]
        ]
        self.statics_preprocessed = KTDataProcessor.get_basic_info(self.data_preprocessed)

        # Q table
        df_new = pd.DataFrame({
            "question_id": map(int, df["question_id"].tolist()),
            "concept_id": map(int, df["concept_id"].tolist())
        })
        Q_table = np.zeros((len(question_ids), len(concept_ids)), dtype=int)
        for question_id, group_info in df_new[["question_id", "concept_id"]].groupby("question_id"):
            correspond_c = pd.unique(group_info["concept_id"]).tolist()
            Q_table[[question_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
        self.Q_table = Q_table

    def uniform_SLP(self):
        df = deepcopy(self.data_preprocessed)
        # school_id按照学生数量重映射
        df["school_id"] = df["school_id"].fillna(-1)
        self.other_info["school_id_map"] = map_user_info(df, "school_id")

        info_name_table = {
            "question_seq": "question_id",
            "correctness_seq": "correctness",
            "time_seq": "timestamp",
            "mode_seq": "interaction_type",
            "answer_score_seq": "answer_score"
        }

        id_keys = list(set(df.columns) - set(info_name_table.values()) - {"concept_id"})
        seq_keys = deepcopy(CONSTANT.datasets_seq_keys()["SLP"])
        seqs = []
        user_id_map = {}
        for i, user_id in enumerate(pd.unique(df["user_id"])):
            user_id_map[user_id] = i
            user_data = df[df["user_id"] == user_id]
            user_data = user_data.sort_values(by=["timestamp", "order"])
            object_data = {seq_key: [] for seq_key in seq_keys}
            for k in id_keys:
                if k == "user_id":
                    object_data[k] = i
                else:
                    object_data[k] = user_data.iloc[0][k]
            for _, row_data in user_data.iterrows():
                for seq_key in seq_keys:
                    if seq_key not in ["answer_score_seq"]:
                        object_data[seq_key].append(int(row_data[info_name_table[seq_key]]))
                    else:
                        object_data[seq_key].append(row_data[info_name_table[seq_key]])
            object_data["seq_len"] = len(object_data["correctness_seq"])
            seqs.append(object_data)
        self.data_uniformed = list(filter(lambda item: 2 <= item["seq_len"], seqs))
        self.user_id_map = pd.DataFrame({
            "original_id": user_id_map.keys(),
            "mapped_id": user_id_map.values()
        })

    def get_all_id_maps(self):
        self.other_info.update({
            "question_id_map": self.question_id_map,
            "concept_id_map": self.concept_id_map,
            "user_id_map": self.user_id_map
        })
        return self.other_info
