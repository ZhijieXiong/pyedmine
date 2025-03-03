import os
import random
import shutil


def key_str2int(obj):
    obj_new = {}
    for k in obj:
        obj_new[int(k)] = obj[k]
    return obj_new


def merge_school(all_school_info, min_school_user=100, min_mean_seq_len=20):
    # 将人数太少的学校合并起来，并剔除一些学校（序列太短）不能被选为测试集
    result = {}
    i = 0
    not_test = []
    not_test_merged = []
    for school_id, school_info in all_school_info.items():
        if school_id == -1:
            continue
        result.setdefault(i, [])
        num_user = school_info["num_user"]
        num_interaction = school_info["num_interaction"]
        mean_num_interaction = num_interaction / num_user

        if mean_num_interaction < min_mean_seq_len:
            not_test.append(school_id)
            continue

        if len(result[i]) == 0:
            # 如果当前合并id下没有学校，直接将下一个学校加入进来
            result[i].append(school_id)
            if num_user > min_school_user:
                i += 1
            continue

        total_num_user = 0
        for selected_school in result[i]:
            total_num_user += all_school_info[selected_school]["num_user"]
        result[i].append(school_id)
        if (total_num_user + num_user) > min_school_user:
            i += 1

    if len(result[i]) > 0:
        i += 1

    for school_id in not_test:
        result.setdefault(i, [])
        num_user = all_school_info[school_id]["num_user"]

        if len(result[i]) == 0:
            not_test_merged.append(i)
            # 如果当前合并id下没有学校，直接将下一个学校加入进来
            result[i].append(school_id)
            if num_user > min_school_user:
                i += 1
            continue

        total_num_user = 0
        for selected_school in result[i]:
            total_num_user += all_school_info[selected_school]["num_user"]
        result[i].append(school_id)
        if (total_num_user + num_user) > min_school_user:
            i += 1

    return result, not_test_merged


def refine_merged_school_info(merged_schools, all_school_info):
    merged_schools_new = {}
    for merged_id, original_ids in merged_schools.items():
        merged_schools_new[merged_id] = {
            "school_ids": original_ids,
            "num_user": 0,
            "num_interaction": 0
        }
        for ori_id in original_ids:
            merged_schools_new[merged_id]["num_user"] += all_school_info[ori_id]["num_user"]
            merged_schools_new[merged_id]["num_interaction"] += all_school_info[ori_id]["num_interaction"]

    return merged_schools_new


def split(merged_school_info, not_test_merged_schools, train_ratio=0.8):
    total_num_user = 0
    total_num_interaction = 0
    for school_info in merged_school_info.values():
        total_num_user += school_info["num_user"]
        total_num_interaction += school_info["num_interaction"]

    not_test_num_user = 0
    not_test_num_interaction = 0
    for not_test_id in not_test_merged_schools:
        not_test_num_user += merged_school_info[not_test_id]["num_user"]
        not_test_num_interaction += merged_school_info[not_test_id]["num_interaction"]

    not_test_ratio = not_test_num_interaction / total_num_interaction
    trainable_schools = list(set(merged_school_info.keys()) - set(not_test_merged_schools))
    num_school_test = int(len(trainable_schools) * ((1 - train_ratio) / (1 - not_test_ratio)))
    if random.random() > 0.5:
        num_school_test += 1
    random.shuffle(trainable_schools)
    schools_test = trainable_schools[:num_school_test]

    total_train_num_user = 0
    total_train_num_interaction = 0
    schools_train = list(set(trainable_schools) - set(schools_test)) + not_test_merged_schools
    for train_id in schools_train:
        total_train_num_user += merged_school_info[train_id]["num_user"]
        total_train_num_interaction += merged_school_info[train_id]["num_interaction"]

    total_test_num_user = 0
    total_test_num_interaction = 0
    for test_id in schools_test:
        total_test_num_user += merged_school_info[test_id]["num_user"]
        total_test_num_interaction += merged_school_info[test_id]["num_interaction"]

    return schools_test, (total_train_num_user, total_test_num_user), (total_train_num_interaction, total_test_num_interaction)


def split_data(data_all, test_merged_schools, merged_school_info):
    test_schools_original_id = []
    for merged_id in test_merged_schools:
        test_schools_original_id.extend(merged_school_info[merged_id]["school_ids"])

    data_train_iid = list(filter(lambda x: x["school_id"] not in test_schools_original_id, data_all))
    data_test_ood = list(filter(lambda x: x["school_id"] in test_schools_original_id, data_all))

    return data_train_iid, data_test_ood


def copy_file(src_dir, dst_dir, dataset_name):
    all_files_in_setting = os.listdir(src_dir)
    for file_name in all_files_in_setting:
        if dataset_name in file_name:
            src_file_path = os.path.join(src_dir, file_name)
            dst_file_path = os.path.join(dst_dir, file_name)
            shutil.copy2(src_file_path, dst_file_path)


def rename_file(setting_dir):
    # [dataset_name]_[multi | single]_train.txt --> [dataset_name]_[multi | single]_train_fold_0.txt
    # [dataset_name]_[multi | single]_test_iid.txt --> [dataset_name]_[multi | single]_valid_fold_0.txt
    # [dataset_name]_[multi | single]_test_ood.txt --> [dataset_name]_[multi | single]_test.txt
    all_files_in_setting = os.listdir(setting_dir)
    for file_name in all_files_in_setting:
        if "test_iid.txt" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("test_iid", "valid_fold_0"))
            os.rename(old_name, new_name)
            continue
        if "test_ood.txt" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("_ood", ""))
            os.rename(old_name, new_name)
            continue
        if "train.txt" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("train", "train_fold_0"))
            os.rename(old_name, new_name)
            continue


def copy_file_all(src_dir, dst_dir):
    all_files_in_setting = os.listdir(src_dir)
    for file_name in all_files_in_setting:
        src_file_path = os.path.join(src_dir, file_name)
        dst_file_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_file_path, dst_file_path)


def reverse_train_valid_name(setting_dir):
    all_files_in_setting = os.listdir(setting_dir)
    for file_name in all_files_in_setting:
        if "valid" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            tmp_new_name = os.path.join(setting_dir, file_name.replace("valid_fold_0", "test_tmp"))
            os.rename(old_name, tmp_new_name)
        if "test" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            tmp_new_name = os.path.join(setting_dir, file_name.replace("test", "valid_fold_0_tmp"))
            os.rename(old_name, tmp_new_name)

    all_files_in_setting = os.listdir(setting_dir)
    for file_name in all_files_in_setting:
        if "valid" in file_name or "test" in file_name:
            old_name = os.path.join(setting_dir, file_name)
            new_name = os.path.join(setting_dir, file_name.replace("_tmp", ""))
            os.rename(old_name, new_name)
