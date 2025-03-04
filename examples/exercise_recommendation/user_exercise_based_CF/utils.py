def delete_test_data(users_data):
    # 只能用训练集数据和验证集数据计算习题相似矩阵
    for user_data in users_data: 
        valid_end_idx = user_data["valid_end_idx"]
        user_data["seq_len"] = valid_end_idx
        for k, v in user_data.items():
            if type(v) is list:
                seq_len = len(v)
                user_data[k] = v[:valid_end_idx] + [0] * (seq_len - valid_end_idx)
