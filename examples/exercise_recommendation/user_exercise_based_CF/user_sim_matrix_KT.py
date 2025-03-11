import argparse
import os
import numpy as np

from config import config_roster
from utils import delete_test_data

from edmine.roster.DLKTRoster import DLKTRoster
from edmine.utils.data_io import read_kt_file
from edmine.utils.calculate import cosine_similarity_matrix, pearson_similarity


def data2batches(data, batch_size):
    batches = []
    batch = []
    for item_data in data:
        if len(batch) < batch_size:
            batch.append(item_data)
        else:
            batches.append(batch)
            batch = [item_data]
    if len(batch) > 0:
        batches.append(batch)
    return batches


def get_mlkc(roster, data_batches):
    mlkc_all_user = {}
    for batch in data_batches:
        users_mlkc = roster.get_knowledge_state(batch).detach().cpu().numpy()
        for i, user_mlkc in enumerate(users_mlkc):
            for j, mlkc in enumerate(user_mlkc):
                users_mlkc[i][j] = round(mlkc, 2)
        user_ids = [item_data["user_id"] for item_data in batch]
        for user_id, user_mlkc in zip(user_ids, users_mlkc):
            mlkc_all_user[user_id] = user_mlkc
    return mlkc_all_user


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据配置
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--user_data_file_name", type=str, default="assist2009_user_data.txt")
    # 加载KT模型
    parser.add_argument("--model_dir_name", type=str, help="模型文件夹名",
                        default=r"DKT@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-03@23-46-42")
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # batch size
    parser.add_argument("--batch_size", type=int, default=256)
    # 相似度选择
    parser.add_argument("--similarity", type=str, default="cossim", choices=("cossim", "pearson_corr"))
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_roster(params)
    kt_roster = DLKTRoster(global_params, global_objects)

    file_manager = global_objects["file_manager"]
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    users_data = read_kt_file(os.path.join(setting_dir, f"{params['dataset_name']}_user_data.txt"))
    delete_test_data(users_data)
    # 使用知识追踪训练集和验证集的数据找相似用户
    kt_setting_dir = file_manager.get_setting_dir("pykt_setting")
    kt_train_data = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_train.txt"))
    kt_valid_data = read_kt_file(os.path.join(kt_setting_dir, f"{params['dataset_name']}_valid.txt"))
    users_data += kt_train_data + kt_valid_data
    user_ids = list(map(lambda x: x["user_id"], users_data))
    num_user = max(user_ids) + 1

    batches_train = data2batches(users_data, params["batch_size"])
    mlkc = get_mlkc(kt_roster, batches_train)
    num_concept = len(mlkc[list(mlkc.keys())[0]])
    user_concept_matrix = np.zeros((num_user, num_concept))
    for user_id, user_data in mlkc.items():
        user_concept_matrix[user_id] = mlkc[user_id]

    if params["similarity"] == "cossim":
        user_similarity = cosine_similarity_matrix(user_concept_matrix, axis=1)
    elif params["similarity"] == "pearson_corr":
        user_similarity = np.zeros((num_user, num_user))
        for i in range(num_user):
            for j in range(num_user):
                si = user_concept_matrix[i, :]
                sj = user_concept_matrix[j, :]
                user_similarity[i][j] = pearson_similarity(si, sj)
    else:
        raise NotImplementedError(f'{params["similarity"]} is not implemented')
    
    save_path = os.path.join(setting_dir,
                             f"{params['dataset_name']}_user_sim_mat_{params['similarity']}_{params['model_dir_name']}.npy")
    np.save(save_path, user_similarity)