import warnings
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, mean_absolute_error


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def get_kt_metric(y_true, y_score):
    assert len(y_true) == len(y_score), "len of y_true and len of y_score must be equal"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if (len(y_true) == 0):
            AUC = -1.
        else:
            AUC = roc_auc_score(y_true, y_score)
        if math.isnan(AUC):
            AUC = -1.
    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    if (len(y_true) == 0):
        return {
            "AUC": -1.,
            "ACC": -1.,
            "MAE": -1.,
            "RMSE": -1.
        }
    else:
        return {
            "AUC": AUC,
            "ACC": accuracy_score(y_true=y_true, y_pred=y_pred),
            "MAE": mean_absolute_error(y_true=y_true, y_pred=y_score),
            "RMSE": root_mean_squared_error(y_true=y_true, y_pred=y_score)
        }


def core_metric(predict_score, ground_truth, question_ids, allow_replace=True):
    question_ids_ = np.unique(question_ids)
    predict_score_balanced = []
    ground_truth_balanced = []

    for q_id in tqdm(question_ids_, desc=f"calculate core metric, {'repeated' if allow_replace else 'non-repeated'}"):
        predict_score4q_id = predict_score[question_ids == q_id]
        ground_truth4q_id = ground_truth[question_ids == q_id]
        num_right = np.sum(ground_truth4q_id == 1)
        num_wrong = np.sum(ground_truth4q_id == 0)

        if num_right == 0 or num_wrong == 0:
            continue

        # 从label为1和0的测试数据中随机选相同数量（官方提供的代码上来看，是允许重复选取的）
        if allow_replace:
            num_balance = (num_wrong + num_right) // 2
        else:
            num_balance = min(num_wrong, num_right)
        index_right = np.random.choice(np.where(ground_truth4q_id == 1)[0], num_balance, replace=allow_replace)
        index_wrong = np.random.choice(np.where(ground_truth4q_id == 0)[0], num_balance, replace=allow_replace)
        index_balanced = list(index_right) + list(index_wrong)
        predict_score_balanced.append(predict_score4q_id[index_balanced])
        ground_truth_balanced.append(ground_truth4q_id[index_balanced])

    predict_score_balanced = np.concatenate(predict_score_balanced)
    ground_truth_balanced = np.concatenate(ground_truth_balanced)

    return get_kt_metric(ground_truth_balanced, predict_score_balanced)
