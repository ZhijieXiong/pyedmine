import math
import numpy as np


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
    mlkc_all_user = []
    for batch in data_batches:
        users_mlkc = roster.get_knowledge_state(batch).detach().cpu().numpy().tolist()
        for i, user_mlkc in enumerate(users_mlkc):
            for j, mlkc in enumerate(user_mlkc):
                users_mlkc[i][j] = round(mlkc, 2)
        user_ids = [item_data["user_id"] for item_data in batch]
        for user_id, user_mlkc in zip(user_ids, users_mlkc):
            mlkc_all_user.append((user_id, user_mlkc))
    return mlkc_all_user


def save_data(data_path, data_values):
    with open(data_path, "w") as f:
        for data_value in data_values:
            user_id, target_value = data_value
            f.write(f"{user_id}:" + ','.join(list(map(str, target_value))) + "\n")


def get_last_frkc(user_data, q2c, num_concept, theta):
    seq_len = user_data["seq_len"]
    if "time_seq" in user_data:
        time_seq = user_data["time_seq"][:seq_len]
    else:
        time_seq = list(range(seq_len))
    last_time = time_seq[-1]
    question_seq = user_data["question_seq"][:seq_len]
    # 以小时为时间单位
    delta_t_max = (last_time - time_seq[0]) / 60
    delta_t_from_last = [delta_t_max] * num_concept
    for t, q in zip(time_seq[:-1], question_seq[:-1]):
        cs = q2c[q]
        delta_t = (last_time - t) / 60
        for c in cs:
            delta_t_from_last[c] = delta_t
    frkc = []
    for delta_t in delta_t_from_last:
        frkc.append(1 - math.exp(-theta * delta_t))
    return frkc


def get_efr(user_id, frkc, q2c):
    efr = []
    for q, cs in q2c.items():
        efr.append(round(sum([frkc[c] for c in cs]) / len(cs), 2))
    return user_id, efr


def cosine_similarity(list1, list2):
    # 将列表转换为NumPy数组
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # 计算点积
    dot_product = np.dot(arr1, arr2)

    # 计算向量的模
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)

    # 计算余弦相似度
    if norm_arr1 == 0 or norm_arr2 == 0:
        return 0.0  # 避免除以零的情况
    cosine_sim = dot_product / (norm_arr1 * norm_arr2)

    return cosine_sim


def get_recommended_exercises(q2c, q_table, mlkc_, pkc_, efr_, delta1=0.7, delta2=0.7, top_n=10):
    scores = []
    for q, cs in q2c.items():
        score1 = 1
        for c in cs:
            score1 *= mlkc_[c]
        score1 = math.pow(delta1 - score1, 2)
        score2 = math.pow(cosine_similarity(q_table[q], pkc_), 2)
        score3 = math.pow(delta2 - efr_[q], 2)
        scores.append((q, score1 + score2 + score3))
    return list(map(lambda x: x[0], sorted(scores, key=lambda x: x[1])))[:top_n]


def save_triples(triples_path, mlkc_all, pkc_all, efr_all, rec_exs=None):
    with open(triples_path, "w") as f:
        for user_id, mlkc_ in mlkc_all.items():
            for k, m in enumerate(mlkc_):
                f.write(f"kc{k}\tmlkc{m}\tuid{user_id}\n")
        for user_id, pkc_ in pkc_all.items():
            for k, p in enumerate(pkc_):
                f.write(f"kc{k}\tpkc{p}\tuid{user_id}\n")
        for user_id, efr_ in efr_all.items():
            for q, e in enumerate(efr_):
                f.write(f"ex{q}\tefr{e}\tuid{user_id}\n")
        if rec_exs is not None:
            for user_id, rec_ex_ in rec_exs.items():
                for q in rec_ex_:
                    f.write(f"uid{user_id}\trec\tex{q}\n")