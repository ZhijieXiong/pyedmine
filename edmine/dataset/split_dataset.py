import random
import os


def kt_select_test_data(data, test_radio, user_id_remap=False):
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


def split_kt_dataset(data, n_fold, test_radio, seed=0):
    """
    选一部分数据做测试集，剩余数据用n折交叉划分为训练集和验证集
    :param test_radio:
    :param data:
    :param n_fold:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_n], [valid_fold_0, ..., valid_fold_n], test)
    """
    random.seed(seed)
    dataset_train_valid, dataset_test = kt_select_test_data(data, test_radio, user_id_remap=True)
    num_train_valid = len(dataset_train_valid)
    num_fold = (num_train_valid // n_fold) + 1
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


def split_cd_dataset(data, n_fold, test_radio, seed=0):
    """
    对每个学生的数据选一部分做测试集，剩余数据用n折交叉划分为训练集和验证集
    :param test_radio:
    :param data:
    :param n_fold:
    :param seed:
    :return: ([train_fold_0, ..., train_fold_n], [valid_fold_0, ..., valid_fold_n], test)
    """
    random.seed(seed)
    dataset_train_valid = []
    dataset_test = []
    i = 0
    for user_data in data:
        num_intercation = user_data["num_interaction"]
        if num_intercation < 10:
            continue

        interaction_data = user_data["all_interaction_data"]
        for interaction in interaction_data:
            interaction["user_id"] = i
        
        interaction_data1 = interaction_data[:int(num_intercation/2)]
        random.shuffle(interaction_data1)
        num1 = len(interaction_data1)
        dataset_test.extend(interaction_data1[:int(num1 * test_radio)])
        dataset_train_valid.extend(interaction_data1[int(num1 * test_radio):])

        interaction_data2 = interaction_data[int(num_intercation/2):]
        random.shuffle(interaction_data2)
        num2 = len(interaction_data2)
        dataset_test.extend(interaction_data2[:int(num2 * test_radio)])
        dataset_train_valid.extend(interaction_data2[int(num2 * test_radio):])

        i += 1

    num_train_valid = len(dataset_train_valid)
    num_fold = (num_train_valid // n_fold) + 1
    dataset_folds = [dataset_train_valid[num_fold * fold: num_fold * (fold + 1)] for fold in range(n_fold)]
    result = ([], [], dataset_test, i)
    for i in range(n_fold):
        fold_valid = i
        result[1].append(dataset_folds[fold_valid])
        folds_train = set(range(n_fold)) - {fold_valid}
        data_train = []
        for fold in folds_train:
            data_train += dataset_folds[fold]
        result[0].append(data_train)

    return result


def n_fold_split(dataset_name, data, setting, file_manager, write_func, task_name="kt"):
    n_fold = setting["n_fold"]
    test_radio = setting["test_radio"]
    setting_name = setting["name"]
    setting_dir = file_manager.get_setting_dir(setting_name)

    assert n_fold > 1, "n_fold must > 1"

    if task_name == "kt":
        datasets_train, datasets_valid, dataset_test = split_kt_dataset(data, n_fold, test_radio)
    elif task_name == "cd":
        datasets_train, datasets_valid, dataset_test, num_user = split_cd_dataset(data, n_fold, test_radio)
        with open(os.path.join(setting_dir, f"{dataset_name}_statics.txt"), "w") as f:
            f.write(f"num of user: {num_user}\n")
    else:
        raise NotImplementedError(f"n fold split for `{task_name}` is not implemented")
    names_train = [f"{dataset_name}_train_fold_{fold}.txt" for fold in range(n_fold)]
    names_valid = [f"{dataset_name}_valid_fold_{fold}.txt" for fold in range(n_fold)]
    for fold in range(n_fold):
        write_func(datasets_train[fold], os.path.join(setting_dir, names_train[fold]))
        write_func(datasets_valid[fold], os.path.join(setting_dir, names_valid[fold]))
    write_func(dataset_test, os.path.join(setting_dir, f"{dataset_name}_test.txt"))

    # 用于调参的数据集
    train4tuning = []
    valid4tuning = []
    for n in range(n_fold):
        random.shuffle(datasets_valid[n])
        num_valid = int(len(datasets_valid[n]) * 0.3)
        valid4tuning.extend(datasets_valid[n][:num_valid])
        train4tuning.extend(datasets_valid[n][num_valid:])

    write_func(train4tuning, os.path.join(setting_dir, f"{dataset_name}_train.txt"))
    write_func(valid4tuning, os.path.join(setting_dir, f"{dataset_name}_valid.txt"))
