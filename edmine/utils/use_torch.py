import torch
import random
import numpy as np

from edmine.utils.check import check_q_table


def parse_q_table(q_table: np.ndarray, device: str):
    """
    Processes a question-concept relationship matrix (q_table) to generate:
    A table mapping each question to its associated concepts.
    A mask table to handle padding for questions with fewer concepts than the maximum.
    :param q_table: A 2D NumPy array representing the question-concept relationship, where rows correspond to questions and columns correspond to concepts. A value of 1 indicates a relationship between a question and a concept.
    :param device: The device (e.g., CPU or GPU) where the output tensors should be allocated.
    :return: q2c_table, : A tensor of shape (num_questions, num_max_c_in_q) where each row contains the concept IDs associated with a question. Padding is used for questions with fewer concepts than num_max_c_in_q. || q2c_mask_table, A tensor of shape (num_questions, num_max_c_in_q) where each row contains a mask indicating valid concept IDs (1) and padding (0).
    """
    check_q_table(q_table)
    q2c_table = []
    q2c_mask_table = []
    num_max_c_in_q = np.max(np.sum(q_table, axis=1))
    num_question = q_table.shape[0]
    for i in range(num_question):
        cs = np.argwhere(q_table[i] == 1).reshape(-1).tolist()
        pad_len = num_max_c_in_q - len(cs)
        q2c_table.append(cs + [0] * pad_len)
        q2c_mask_table.append([1] * len(cs) + [0] * pad_len)
    q2c_table = torch.tensor(q2c_table).long().to(device)
    q2c_mask_table = torch.tensor(q2c_mask_table).long().to(device)
    return q2c_table, q2c_mask_table


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    return torch.backends.mps.is_available()


def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed, details are ", e)
        pass
    np.random.seed(seed)
    random.seed(seed)

