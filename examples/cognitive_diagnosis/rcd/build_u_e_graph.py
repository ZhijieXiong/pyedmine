import argparse
import os

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_cd_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="ncd_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--train_file_name", type=str, default="assist2009_train_fold_0.txt")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    train_data_path = os.path.join(setting_dir, params["train_file_name"])
    
    train_data = read_cd_file(train_data_path)
    data_statics_path = os.path.join(setting_dir, f"{params['dataset_name']}_statics.txt")
    with open(data_statics_path, "r") as f:
        s = f.readline()
        num_user = int(s.split(":")[1].strip())
    q_table = file_manager.get_q_table(params['dataset_name'])
    q2c = q2c_from_q_table(q_table)
    num_question, num_concept = q_table.shape[0], q_table.shape[1]
    
    u_from_e = '' # e(src) to k(dst)
    e_from_u = '' # k(src) to k(dst)
    for interaction in train_data:
        q_id = interaction['question_id']
        user_id = interaction['user_id']
        c_ids = q2c[q_id]
        for _ in c_ids:
            u_from_e += str(q_id) + '\t' + str(user_id + num_question) + '\n'
            e_from_u += str(user_id + num_question) + '\t' + str(q_id) + '\n'
            
    graph_dir = os.path.join(setting_dir, "RCD")
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
        
    with open(os.path.join(graph_dir, f"u_from_e_{params['train_file_name']}"), 'w') as f:
        f.write(u_from_e)
    with open(os.path.join(graph_dir, f"e_from_u_{params['train_file_name']}"), 'w') as f:
        f.write(e_from_u)
