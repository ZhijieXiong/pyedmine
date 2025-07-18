import argparse

from config import config_roster

from edmine.roster.DLCDRoster import DLCDRoster
from edmine.utils.data_io import read_kt_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_name", type=str, 
                        default=r"NCD@@ncd_setting@@assist2009_train_fold_0@@seed_0@@2025-03-11@01-46-58")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_roster(params)
    roster = DLCDRoster(global_params, global_objects)
    data = read_kt_file("/data/dataset/settings/ER_offline_setting/assist2009_user_data.txt")
    user_concept_mastery_level = roster.get_knowledge_state([0,1,2,3])
    print(user_concept_mastery_level[0])
