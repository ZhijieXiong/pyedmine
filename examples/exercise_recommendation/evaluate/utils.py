import os

from edmine.utils.data_io import read_id_map_kg4ex


def get_model_info(model_dir_name):
    model_info = model_dir_name.split("@@")
    model_name, setting_name, train_file_name = model_info[0], model_info[1], model_info[2]
    return model_name, setting_name, train_file_name


def config_kg4ex(local_params, global_objects, setting_name):
    dataset_name = local_params["dataset_name"]
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    kg4ex_dir = os.path.join(setting_dir, "KG4EX")
    global_objects["dataset"]["entity2id"] = read_id_map_kg4ex(os.path.join(kg4ex_dir, f'{dataset_name}_entities_kg4ex.dict'))
    # 存储relations
    relations_path = os.path.join(kg4ex_dir, "relations_kg4ex.dict")
    if not os.path.exists(relations_path):
        scores = [round(i * 0.01, 2) for i in range(101)]
        with open(relations_path, "w") as fs:
            for i, s in enumerate(scores):
                fs.write(f"{i}\tmlkc{s}\n")
            for i, s in enumerate(scores):
                fs.write(f"{i + 101}\tpkc{s}\n")
            for i, s in enumerate(scores):
                fs.write(f"{i + 202}\tefr{s}\n")
            fs.write("303\trec")
    global_objects["dataset"]["relation2id"] = read_id_map_kg4ex(os.path.join(kg4ex_dir, 'relations_kg4ex.dict'))