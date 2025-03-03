import random
import os


def kt_select_test_data1(data, test_radio, user_id_remap=False):
    if user_id_remap:
        for i, item in enumerate(data):
            item["user_id"] = i

    test_data = []
    train_valid_data = []

    # 按长度序列分层
    # 默认序列经过补零，长度一致
    max_seq_len = len(data[0]["correctness_seq"])
    layer_seq_len = max_seq_len // 4
    data_layer_by_len = {i: [] for i in range(4)}
    for item in data:
        seq_len = item["seq_len"]
        idx = min(3, seq_len // layer_seq_len)
        data_layer_by_len[idx].append(item)

    # 再按正确率分层
    for seq_len_data in data_layer_by_len.values():
        data_layer_by_acc = {i: [] for i in range(4)}
        for item in seq_len_data:
            seq_len = item["seq_len"]
            correctness_seq = item["correctness_seq"][:seq_len]
            acc = sum(correctness_seq) / seq_len
            idx = min(3, int(acc // 0.25))
            data_layer_by_acc[idx].append(item)

        for acc_data in data_layer_by_acc.values():
            random.shuffle(acc_data)
            num_data = len(acc_data)
            num_train_valid = int(num_data * (1 - test_radio))
            train_valid_data += acc_data[:num_train_valid]
            test_data += acc_data[num_train_valid:]

    return train_valid_data, test_data


def split(data, n_fold, test_radio, seed=0):
    """
    选一部分数据做测试集，剩余数据用n折交叉划分为训练集和验证集
    :param test_radio:
    :param data:
    :param n_fold:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_n], [valid_fold_0, ..., valid_fold_n], test)
    """
    random.seed(seed)
    random.shuffle(data)
    num_all = len(data)
    num_train_valid = int(num_all * (1 - test_radio))
    num_fold = (num_train_valid // n_fold) + 1

    dataset_train_valid, dataset_test = kt_select_test_data1(data, test_radio, user_id_remap=True)
    dataset_folds = [dataset_train_valid[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    result = ([], [], dataset_test)
    for i in range(n_fold):
        fold_valid = i
        result[1].append(dataset_folds[fold_valid])
        folds_train = set(range(n_fold)) - {fold_valid}
        data_train = []
        for fold in folds_train:
            data_train += dataset_folds[fold]
        result[0].append(data_train)

    return result


def n_fold_split(dataset_name, data, setting, file_manager, write_func):
    n_fold = setting["n_fold"]
    test_radio = setting["test_radio"]
    setting_name = setting["name"]

    assert n_fold > 1, "n_fold must > 1"

    datasets_train, datasets_valid, dataset_test = split(data, n_fold, test_radio)
    names_train = [f"{dataset_name}_train_fold_{fold}.txt" for fold in range(n_fold)]
    names_valid = [f"{dataset_name}_valid_fold_{fold}.txt" for fold in range(n_fold)]

    setting_dir = file_manager.get_setting_dir(setting_name)
    for fold in range(n_fold):
        write_func(datasets_train[fold], os.path.join(setting_dir, names_train[fold]))
        write_func(datasets_valid[fold], os.path.join(setting_dir, names_valid[fold]))

    write_func(dataset_test, os.path.join(setting_dir, f"{dataset_name}_test.txt"))

    # 用于调参，如果数据集非常大，只用部分数据调参
    dataset_train_valid = datasets_train[0] + datasets_valid[0]
    random.shuffle(dataset_train_valid)
    dataset_train_valid = dataset_train_valid[:20000]
    valid_radio = 0.2
    num_train = int(len(dataset_train_valid) * (1 - valid_radio))
    write_func(dataset_train_valid[:num_train], os.path.join(setting_dir, f"{dataset_name}_train.txt"))
    write_func(dataset_train_valid[num_train:], os.path.join(setting_dir, f"{dataset_name}_valid.txt"))
