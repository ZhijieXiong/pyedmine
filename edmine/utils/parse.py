import argparse
import ast
import numpy as np

from edmine.utils.check import check_q_table


def c2q_from_q_table(q_table: np.ndarray) -> dict[int, list[int]]:
    """
    Converts a question-concept matrix (q_table) into a dictionary mapping each concept to its associated questions.
    :param q_table: A 2D NumPy array representing the question-concept relationship, where rows correspond to questions and columns correspond to concepts. A value of 1 indicates a relationship between a question and a concept.
    :return: A dictionary where each key is a concept ID, and the value is a list of question IDs linked to that concept.
    """
    check_q_table(q_table)
    return {i: np.argwhere(q_table[:, i] == 1).reshape(-1).tolist() for i in range(q_table.shape[1])}


def q2c_from_q_table(q_table: np.ndarray) -> dict[int, list[int]]:
    """
    Converts a question-concept matrix (q_table) into a dictionary mapping each question to its associated concepts.
    :param q_table: A 2D NumPy array representing the question-concept relationship, where rows correspond to questions and columns correspond to concepts. A value of 1 indicates a relationship between a question and a concept.
    :return: A dictionary where each key is a question ID, and the value is a list of concept IDs linked to that concept.
    """
    check_q_table(q_table)
    return {i: np.argwhere(q_table[i] == 1).reshape(-1).tolist() for i in range(q_table.shape[0])}


def get_kt_data_statics(kt_data: list[dict], q_table: np.ndarray) -> dict:
    """
    Computes key statistics for a knowledge tracing dataset, including the number of sequences, total samples, average sequence length, average question accuracy, and question/concept sparsity.
    :param kt_data:
    :param q_table: A 2D NumPy array representing the question-concept relationship, where rows correspond to questions and columns correspond to concepts. A value of 1 indicates a relationship between a question and a concept.
    :return: A dictionary containing the following statistics: `num_seq`, `num_sample`, `ave_seq_len`, `ave_que_acc`, `que_sparsity`, `concept_sparsity`
    """
    check_q_table(q_table)

    num_question, num_concept = q_table.shape

    num_seq = len(kt_data)
    num_sample = sum(list(map(lambda x: x["seq_len"], kt_data)))
    ave_seq_len = round(num_sample/num_seq, 2)
    num_right = 0
    for item_data in kt_data:
        seq_len = item_data["seq_len"]
        num_right += sum(item_data["correctness_seq"][:seq_len])
    ave_que_acc = round(num_right / num_sample, 4)

    U = len(kt_data)
    Q = num_question
    C = num_concept
    q2c = q2c_from_q_table(q_table)
    user_que_mat = np.zeros((U, Q))
    user_concept_mat = np.zeros((U, C))
    for u, item_data in enumerate(kt_data):
        seq_len = item_data["seq_len"]
        for j in range(seq_len):
            q = item_data["question_seq"][j]
            user_que_mat[u][q] = 1
            cs = q2c[q]
            for c in cs:
                user_concept_mat[u][c] = 1
    que_sparsity = round(1 - np.sum(user_que_mat) / (U * Q), 4)
    concept_sparsity = round(1 - np.sum(user_concept_mat) / (U * C), 4)

    return {
        "num_seq": num_seq,
        "num_sample": num_sample,
        "ave_seq_len": ave_seq_len,
        "ave_que_acc": ave_que_acc,
        "que_sparsity": que_sparsity,
        "concept_sparsity": concept_sparsity,
    }


def str2bool(v):
    """
    Converts a string to a boolean value based on common representations of True and False.
    :param v: A string representing a boolean value. Accepted values for True include "yes", "true", "t", "y", and "1". Accepted values for False include "no", "false", "f", "n", and "0".
    :return: True of False
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_keys_from_kt_data(kt_data: list[dict]) -> tuple:
    """
    Categorizes the keys of dictionaries in a uniformly formatted list into:
    ID keys: Keys associated with single values (non-list values).
    Sequence keys: Keys associated with lists (sequential data).
    :param kt_data: A list of dictionaries where all dictionaries have the same structure. Each dictionary represents an item in the dataset, and the keys represent attributes of the item.
    :return: (id_keys, seq_keys)
    """
    item_data = kt_data[0]
    id_keys = []
    for k in item_data.keys():
        if type(item_data[k]) is not list:
            id_keys.append(k)
    seq_keys = list(set(item_data.keys()) - set(id_keys))
    return id_keys, seq_keys


def params2str_tool(param):
    if isinstance(param, set) or isinstance(param, list) or type(param) is bool:
        return str(param)
    elif type(param) in (int, float, str):
        return param
    else:
        return "not transform"


def params2str(params):
    """
    Converts a dictionary of parameters into a JSON-serializable format.
    :param params:
    :return:
    """
    params_json = {}
    for k, v in params.items():
        if type(v) is not dict:
            params_json[k] = params2str_tool(v)
        else:
            params_json[k] = params2str(v)
    return params_json


def is_valid_eval_string(in_str):
    try:
        ast.literal_eval(in_str)
        return True
    except (SyntaxError, ValueError):
        return False


def str_dict2params_tool(param):
    if is_valid_eval_string(param):
        return eval(param)
    else:
        return param


def str_dict2params(str_dict):
    params = {}
    for k, v in str_dict.items():
        if type(v) is not dict:
            params[k] = str_dict2params_tool(v)
        else:
            params[k] = str_dict2params(v)
    return params


def kt_data2cd_data(kt_data, useful_keys={"use_time_seq": "use_time"}):
    data4cd = []
    for item_data in kt_data:
        user_data = {
            "user_id": item_data["user_id"],
            "num_interaction": item_data["seq_len"],
            "all_interaction_data": []
        }
        for i in range(item_data["seq_len"]):
            interaction_data = {
                "question_id": item_data["question_seq"][i],
                "correctness": item_data["correctness_seq"][i]
            }
            for kt_key, cd_key in useful_keys.items():
                if kt_key in item_data:
                    interaction_data[cd_key] = item_data[kt_key][i]
            user_data["all_interaction_data"].append(interaction_data)
        data4cd.append(user_data)

    return data4cd