from edmine.utils.parse import get_keys_from_kt_data


def truncate2one_seq(data_uniformed, min_seq_len=2, max_seq_len=200, from_start=True):
    """
    截断数据，取最前面或者最后面一段，不足的在后面补0
    :param data_uniformed:
    :param from_start:
    :param min_seq_len:
    :param max_seq_len:
    :return:
    """
    data_uniformed = list(filter(lambda item: min_seq_len <= item["seq_len"], data_uniformed))
    result = []
    id_keys, seq_keys = get_keys_from_kt_data(data_uniformed)
    for item_data in data_uniformed:
        item_data_new = {key: item_data[key] for key in id_keys}
        seq_len = item_data["seq_len"]
        start_index, end_index = 0, seq_len
        if seq_len > max_seq_len and from_start:
            end_index = max_seq_len
        if seq_len > max_seq_len and not from_start:
            start_index = end_index - max_seq_len
        pad_len = max_seq_len - end_index + start_index
        for k in seq_keys:
            item_data_new[k] = item_data[k][start_index:end_index] + [0] * pad_len
        item_data_new["seq_len"] = end_index - start_index
        item_data_new["mask_seq"] = [1] * item_data_new["seq_len"] + \
                                    [0] * (max_seq_len - item_data_new["seq_len"])
        result.append(item_data_new)
    return result


def truncate2multi_seq_(item_data, seq_keys, id_keys, max_seq_len):
    """
    将一个用户的数据进行常规处理，即截断补零（truncate_and_pad1用于处理单知识点的数据，包括有知识点信息和无知识点信息的）
    :param item_data:
    :param seq_keys:
    :param id_keys:
    :param max_seq_len:
    :return:
    """
    seq_len = item_data["seq_len"]
    result = []
    if seq_len <= max_seq_len:
        item_data_new = {key: item_data[key] for key in id_keys}
        pad_len = max_seq_len - seq_len
        for k in seq_keys:
            item_data_new[k] = item_data[k][0:seq_len] + [0] * pad_len
        item_data_new["mask_seq"] = [1] * seq_len + [0] * pad_len
        result.append(item_data_new)
    else:
        num_segment = item_data["seq_len"] // max_seq_len
        num_segment = num_segment if (item_data["seq_len"] % max_seq_len == 0) else (num_segment + 1)
        for segment in range(num_segment):
            item_data_new = {key: item_data[key] for key in id_keys}
            start_index = max_seq_len * segment
            if segment == item_data["seq_len"] // max_seq_len:
                # the last segment
                pad_len = max_seq_len - (item_data["seq_len"] % max_seq_len)
                for k in seq_keys:
                    item_data_new[k] = item_data[k][start_index:start_index+(item_data["seq_len"] % max_seq_len)] + [0] * pad_len
                item_data_new["seq_len"] = item_data["seq_len"] % max_seq_len
                item_data_new["mask_seq"] = [1] * (max_seq_len - pad_len) + [0] * pad_len
            else:
                end_index = max_seq_len * (segment + 1)
                for k in seq_keys:
                    item_data_new[k] = item_data[k][start_index:end_index]
                item_data_new["seq_len"] = max_seq_len
                item_data_new["mask_seq"] = [1] * max_seq_len
            result.append(item_data_new)
    return result


def truncate2multi_seq(data_uniformed, min_seq_len=2, max_seq_len=200):
    """
    截断数据，不足补0，多的当新数据
    :param data_uniformed:
    :param min_seq_len:
    :param max_seq_len:
    :return:
    """
    data_uniformed = list(filter(lambda item: min_seq_len <= item["seq_len"], data_uniformed))
    result = []

    id_keys, seq_keys = get_keys_from_kt_data(data_uniformed)
    for item_data in data_uniformed:
        item_data_new = truncate2multi_seq_(item_data, seq_keys, id_keys, max_seq_len)
        result += item_data_new

    result = list(filter(lambda item: min_seq_len <= item["seq_len"], result))
    return result
