import os
import json
import logging
import pandas as pd

from edmine.utils.parse import params2str


def read_csv(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    """
    Loads a CSV file into a Pandas DataFrame, with options to specify useful columns, rename columns, and handle encoding issues. It can also limit the number of rows read from the file.
    :param data_path: Path to the CSV file.
    :param useful_cols: (Optional) List of columns to load from the CSV. If None, all columns are loaded.
    :param rename_dict: (Optional) A dictionary to rename columns in the format {old_name: new_name}.
    :param num_rows: (Optional) Number of rows to read from the CSV. If None, all rows are read.
    :return: A Pandas DataFrame containing the data from the CSV file, with optional column renaming and row limits applied.
    """
    try:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, index_col=False,
                         nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, index_col=False,
                         nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def read_table(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    """
    Loads a TABLE file into a Pandas DataFrame, with options to specify useful columns, rename columns, and handle encoding issues. It can also limit the number of rows read from the file.
    :param data_path: Path to the CSV file.
    :param useful_cols: (Optional) List of columns to load from the CSV. If None, all columns are loaded.
    :param rename_dict: (Optional) A dictionary to rename columns in the format {old_name: new_name}.
    :param num_rows: (Optional) Number of rows to read from the CSV. If None, all rows are read.
    :return: A Pandas DataFrame containing the data from the CSV file, with optional column renaming and row limits applied.
    """
    try:
        df = pd.read_table(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_table(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def read_SLP(data_dir, dataset_name):
    """
    Loads and processes an SLP dataset by combining unit-level data, term-level data, student information, and family information into a single DataFrame.
    :param data_dir: Directory path where the dataset files are stored.
    :param dataset_name: Name of the dataset, which is used to identify the subject-specific files (e.g., "mat").
    :return:
    """
    subject = dataset_name.split("-")[-1]
    unit_path = os.path.join(data_dir, f"unit-{subject}.csv")
    term_path = os.path.join(data_dir, f"term-{subject}.csv")
    student_path = os.path.join(data_dir, "student.csv")
    family_path = os.path.join(data_dir, "family.csv")

    useful_cols = ["student_id", "question_id", "concept", "score", "full_score", "time_access"]
    family_cols = ["student_id", "live_on_campus"]
    student_cols = ["student_id", "gender", "school_id"]

    unit = read_csv(unit_path, useful_cols)
    term = read_csv(term_path, useful_cols)
    student = read_csv(student_path, student_cols)
    family = read_csv(family_path, family_cols)

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

    # live_on_campus和school_type有nan
    return df[["student_id", "question_id", "concept", "score", "full_score", "time_access", "order",
               "live_on_campus", "school_id", "gender", "interaction_type"]]


def read_ednet_kt1(data_dir, num_file=1):
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
            break
        else:
            try:
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
            df["tags"] = df["tags"].map(process_tags)
            dfs.append(df)

    return pd.concat(dfs, axis=0)


def read_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def write_kt_file(data, data_path):
    # id_keys表示序列级别的特征，如user_id, seq_len
    # seq_keys表示交互级别的特征，如question_id, concept_id
    id_keys = []
    seq_keys = []
    for key in data[0].keys():
        if type(data[0][key]) == list:
            seq_keys.append(key)
        else:
            id_keys.append(key)

    # 不知道为什么，有的数据集到这的时候，数据变成float类型了（比如junyi2015，如果预处理部分数据，就是int，但是如果全量数据，就是float）
    id_keys_ = set(id_keys).intersection({"user_id", "school_id", "premium_pupil", "gender", "seq_len", "campus",
                                          "dataset_type", "order"})
    seq_keys_ = set(seq_keys).intersection({"question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq",
                                            "use_time_first_seq", "num_hint_seq", "num_attempt_seq", "age_seq",
                                            "question_mode_seq"})
    for item_data in data:
        for k in id_keys_:
            try:
                item_data[k] = int(item_data[k])
            except ValueError:
                print(f"value of {k} has nan")
        for k in seq_keys_:
            try:
                item_data[k] = list(map(int, item_data[k]))
            except ValueError:
                print(f"value of {k} has nan")

    with open(data_path, "w") as f:
        first_line = ",".join(id_keys) + ";" + ",".join(seq_keys) + "\n"
        f.write(first_line)
        for item_data in data:
            for k in id_keys:
                f.write(f"{item_data[k]}\n")
            for k in seq_keys:
                f.write(",".join(map(str, item_data[k])) + "\n")


def read_kt_file(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        seq_interaction_keys_str = first_line.split(";")
        id_keys_str = seq_interaction_keys_str[0].strip()
        seq_keys_str = seq_interaction_keys_str[1].strip()
        id_keys = id_keys_str.split(",")
        seq_keys = seq_keys_str.split(",")
        keys = id_keys + seq_keys
        num_key = len(keys)
        all_lines = all_lines[1:]
        data = []
        for i, line_str in enumerate(all_lines):
            if i % num_key == 0:
                item_data = {}
            current_key = keys[int(i % num_key)]
            if current_key in ["time_factor_seq", "hint_factor_seq", "attempt_factor_seq", "answer_score_seq"]:
                line_content = list(map(float, line_str.strip().split(",")))
            else:
                line_content = list(map(int, line_str.strip().split(",")))
            if len(line_content) == 1:
                # 说明是序列级别的特征，即user id、seq len、segment index等等
                item_data[current_key] = line_content[0]
            else:
                # 说明是interaction级别的特征，即question id等等
                item_data[current_key] = line_content
            if i % num_key == (num_key - 1):
                data.append(item_data)

    return data


def save_params(global_params, model_root_dir, logger):
    model_dir_name = global_params["trainer_config"]["save_model_dir_name"]
    model_dir = os.path.join(model_root_dir, model_dir_name)
    global_params["save_model_dir"] = model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        raise ValueError(f"{model_root_dir} does not exists")

    params_path = os.path.join(model_dir, "params.json")
    params_json = params2str(global_params)
    write_json(params_json, params_path)

    log_path = os.path.join(model_dir, "train_log.txt")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def read_mlkc_data(f_path):
    """
    Reads data from a file where each line is formatted as id:num1,num2,num3,....
    :param f_path: The path to the file containing the data.
    :return: A dictionary where: Keys are ids (integers). Values are lists of floats representing the data values.
    """
    data = {}
    with open(f_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line_ = line.split(":")
        user_id, data_value = line_[0], line_[1]
        data[int(user_id)] = list(map(float, data_value.split(",")))
    return data


def read_id_map_kg4ex(file_path):
    id_map = dict()
    with open(file_path, "r") as fin:
        for line in fin:
            id_, ele = line.strip().split('\t')
            id_map[ele] = int(id_)
    return id_map


def write_cd_file(data, data_path):
    id_keys = data[0].keys()
    with open(data_path, "w") as f:
        first_line = ",".join(id_keys) + "\n"
        f.write(first_line)
        for interaction_data in data:
            line_str = ""
            for k in id_keys:
                line_str += str(interaction_data[k]) + ","
            f.write(line_str[:-1] + "\n")


def read_cd_file(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        id_keys_str = first_line.strip()
        id_keys = id_keys_str.split(",")
        all_lines = all_lines[1:]
        data = []
        for i, line_str in enumerate(all_lines):
            interaction_data = {}
            line_content = list(map(int, line_str.strip().split(",")))
            for id_key, v in zip(id_keys, line_content):
                interaction_data[id_key] = v
            data.append(interaction_data)

    return data