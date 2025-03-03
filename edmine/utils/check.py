import numpy as np


def check_q_table(q_table: np.ndarray):
    # Check if q_table is a 2D NumPy array
    if q_table.size == 0 or q_table.ndim != 2:
        raise IndexError("Input q_table must be a 2D NumPy array.")

    # Check if q_table contains only 0s and 1s
    if not np.all(np.isin(q_table, [0, 1])):
        raise ValueError("Input q_table must contain only 0s and 1s.")

    rows_check = np.any(q_table == 1, axis=1)
    cols_check = np.any(q_table == 1, axis=0)

    if not (np.all(rows_check) and np.all(cols_check)):
        raise ValueError("Each row and column of the input q_table has at least one value of 1.")


def check_kt_seq_start(seq_start):
    if seq_start < 2:
        raise ValueError(f"seq_start must greater than 1")
