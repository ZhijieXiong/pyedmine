import torch
import os
from edmine.utils.parse import q2c_from_q_table, c2q_from_q_table
from edmine.utils.use_torch import parse_q_table


def config_q_table(local_params, global_params, global_objects):
    dataset_name = local_params["dataset_name"]
    q_table = global_objects["file_manager"].get_q_table(dataset_name)
    config_q_table_(local_params, global_params, global_objects, q_table)
    

def config_q_table_(local_params, global_params, global_objects, q_table):
    local_params["num_question"], local_params["num_concept"] = q_table.shape[0], q_table.shape[1]
    q2c_table, q2c_mask_table = parse_q_table(q_table, global_params["device"])
    global_objects["dataset"] = {
        "q_table": q_table,
        "q_table_tensor": torch.from_numpy(q_table).long().to(global_params["device"]),
        "q2c": q2c_from_q_table(q_table),
        "c2q": c2q_from_q_table(q_table),
        "q2c_transfer_table": q2c_table,
        "q2c_mask_table": q2c_mask_table
    }


def config_sequential_kt_dataset(local_params, global_params):
    setting_name = local_params["setting_name"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    global_params["datasets_config"] = {
        "train": {
            "setting_name": setting_name,
            "file_name": train_file_name,
            "device": global_params["device"],
        },
        "valid": {
            "setting_name": setting_name,
            "file_name": valid_file_name,
            "device": global_params["device"],
        },
    }

def config_cd_dataset(local_params, global_params, global_objects):
    setting_dir = global_objects["file_manager"].get_setting_dir(local_params["setting_name"])
    data_statics_path = os.path.join(setting_dir, f"{local_params['dataset_name']}_statics.txt")
    with open(data_statics_path, "r") as f:
        s = f.readline()
        local_params["num_user"] = int(s.split(":")[1].strip())
    config_sequential_kt_dataset(local_params, global_params)


def config_clkt_dataset(local_params, global_params):
    setting_name = local_params["setting_name"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    global_params["datasets_config"] = {
        "train": {
            "setting_name": setting_name,
            "file_name": train_file_name,
            "device": global_params["device"],
            "num_aug": 2,
            "aug_order": ['mask', 'crop', 'permute', 'replace'],
            "mask_prob": local_params["mask_prob"],
            "replace_prob": local_params["replace_prob"],
            "crop_prob": local_params["crop_prob"],
            "permute_prob": local_params["permute_prob"],
            "hard_neg_prob": 1
        },
        "valid": {
            "setting_name": setting_name,
            "file_name": valid_file_name,
            "device": global_params["device"]
        },
    }



def config_kg4ex_dataset(local_params, global_params):
    setting_name = local_params["setting_name"]
    train_file_name = local_params["train_file_name"]
    valid_file_name = local_params["valid_file_name"]
    global_params["datasets_config"] = {
        "train": {
            "tail": {
                "setting_name": setting_name,
                "file_name": train_file_name,
                "device": global_params["device"],
                "is_train": True,
                "mode": "head-batch",
                "negative_sample_size": local_params["negative_sample_size"]
            },
            "head": {
                "setting_name": setting_name,
                "file_name": train_file_name,
                "device": global_params["device"],
                "is_train": True,
                "mode": "tail-batch",
                "negative_sample_size": local_params["negative_sample_size"]
            }
        },
        "valid": {
            "setting_name": setting_name,
            "file_name": valid_file_name,
            "device": global_params["device"],
            "is_train": False
        },
    }
