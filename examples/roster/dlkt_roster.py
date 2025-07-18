import argparse

from config import config_roster

from edmine.roster.DLKTRoster import DLKTRoster
from edmine.utils.data_io import read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_name", type=str,
                        default=r"DKT@@pykt_setting@@assist2009_train_fold_0@@seed_0@@2025-03-01@21-47-09")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    parser.add_argument("--dataset_name", type=str, default="assist2009", help="for Q table")
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_roster(params)
    roster = DLKTRoster(global_params, global_objects)
    data = read_kt_file("/Users/dream/myProjects/pyedmine/dataset/settings/ER_offline_setting/assist2009_user_data.txt")
    batch_data = data[:4]
    last_concept_mastery_level = roster.get_knowledge_state(batch_data)
