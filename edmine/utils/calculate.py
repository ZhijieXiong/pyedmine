import numpy as np


def tf_idf_from_q_table(q_table):
    N = q_table.shape[0]
    tf = q_table
    idf = np.log(N / q_table.sum(axis=0))
    return tf * np.expand_dims(idf, axis=0)


def cosine_similarity_matrix(arr, axis=0):
    """
    计算行向量或列向量两两之间的余弦相似度矩阵。

    参数:
        axis: axis=0 表示按列计算，axis=1 表示按行计算。

    返回:
        余弦相似度矩阵。
    """
    if axis == 1:
        arr = arr.T  # 转置，将行向量转换为列向量

    # 归一化列向量（除以 L2 范数）
    norm = np.linalg.norm(arr, axis=0, keepdims=True)  # 计算每列的 L2 范数
    arr_normalized = arr / (norm + 1e-8)  # 归一化

    # 计算余弦相似度矩阵
    cosine_sim = arr_normalized.T @ arr_normalized  # 矩阵乘法计算点积

    return cosine_sim


def cosine_similarity(A, B):
    """
    计算矩阵 A 和矩阵 B 的行向量之间的余弦相似度。

    参数:
        A: 形状为 (m, d) 的矩阵，表示 m 个 d 维向量。
        B: 形状为 (n, d) 的矩阵，表示 n 个 d 维向量。

    返回:
        形状为 (m, n) 的余弦相似度矩阵。
    """
    # 归一化 A 和 B 的行向量（除以 L2 范数）
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)  # 计算 A 的每行 L2 范数
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)  # 计算 B 的每行 L2 范数

    A_normalized = A / (A_norm + 1e-8)  # 归一化 A
    B_normalized = B / (B_norm + 1e-8)  # 归一化 B

    # 计算余弦相似度矩阵
    cosine_sim = A_normalized @ B_normalized.T  # 矩阵乘法计算点积

    return cosine_sim


def pearson_similarity(scores_i, scores_j):
    # 提取共同评分的索引
    common_ids = np.where((scores_i >= 0) & (scores_j >= 0))[0]
    if len(common_ids) == 0:
        return 0.0  # 无共同评分用户

    # 提取共同评分
    scores_i = scores_i[common_ids]
    scores_j = scores_j[common_ids]

    # 计算均值和差值
    mean_i = np.mean(scores_i)
    mean_j = np.mean(scores_j)
    diff_i = scores_i - mean_i
    diff_j = scores_j - mean_j

    # 计算分子和分母
    numerator = np.sum(diff_i * diff_j)
    denominator = np.sqrt(np.sum(diff_i ** 2)) * np.sqrt(np.sum(diff_j ** 2))

    if denominator == 0:
        return 0.0  # 避免除以0

    return numerator / denominator
