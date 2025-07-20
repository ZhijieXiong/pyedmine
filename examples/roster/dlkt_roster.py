import argparse
import os

from config import config_roster

from edmine.roster.DLKTRoster import DLKTRoster
from edmine.utils.data_io import read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_name", type=str,
                        default=r"qDKT@@pykt_setting@@assist2009_train@@seed_0@@2025-07-18@20-22-14")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    parser.add_argument("--dataset_name", type=str, default="assist2009", help="for Q table")
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_roster(params)
    roster = DLKTRoster(global_params, global_objects)
    setting_dir = global_objects["file_manager"].get_setting_dir("pykt_setting")
    data = read_kt_file(os.path.join(setting_dir, "assist2009_test.txt"))
    batch_data = data[:4]
    last_knowledge_state = roster.get_knowledge_state(batch_data)
    print(last_knowledge_state)
