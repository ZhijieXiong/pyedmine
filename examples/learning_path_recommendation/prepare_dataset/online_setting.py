import argparse
import os

import config

from edmine.data.FileManager import FileManager
from edmine.utils.data_io import write_kt_file


def extract_shortest_paths(input_file):
    shortest_paths = {}

    # 读取文件
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path = list(map(int, line.split(",")))
            key = (path[0], path[-1])
            if key not in shortest_paths or len(path) < len(shortest_paths[key]):
                shortest_paths[key] = path

    # 写入文件
    return shortest_paths
    # with open(output_file, "w") as f:
    #     for path in shortest_paths.values():
    #         line = delimiter.join(map(str, path))
    #         f.write(line + "\n")


if __name__ == "__main__":
    # 选择所有最短路径作为测试集
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="junyi2015")
    args = parser.parse_args()
    params = vars(args)

    lpr_setting = {
        "name": "LPR_online_setting",
    }

    file_manager = FileManager(config.FILE_MANAGER_ROOT)
    file_manager.add_new_setting(lpr_setting["name"], lpr_setting)
    preprocessed_dir = file_manager.get_preprocessed_dir(params["dataset_name"])
    setting_dir = file_manager.get_setting_dir(lpr_setting["name"])
    
    target_paths = extract_shortest_paths(os.path.join(preprocessed_dir, "pre_path.txt"))
    data = []
    for start_end, concept_path in target_paths.items():
        data.append({
            "start_concept_id": start_end[0],
            "end_concept_id": start_end[1],
            "shortest_path": concept_path,
        })
    write_kt_file(data, os.path.join(setting_dir, f"{params['dataset_name']}_single_goal.txt"))