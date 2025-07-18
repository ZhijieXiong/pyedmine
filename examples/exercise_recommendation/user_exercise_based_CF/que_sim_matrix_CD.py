import argparse
import os
import torch
import numpy as np

from config import config_roster

from edmine.roster.DLCDRoster import DLCDRoster
from edmine.utils.calculate import cosine_similarity_matrix, pearson_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据配置
    parser.add_argument("--setting_name", type=str, default="ER_offline_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    # 加载DC模型
    parser.add_argument("--model_dir_name", type=str, help="模型文件夹名",
                        default="NCD@@CD_setting4ER_offline_setting@@assist2009_train@@seed_0@@2025-03-11@18-08-00")
    parser.add_argument("--model_file_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    # 相似度选择
    parser.add_argument("--similarity", type=str, default="cossim", choices=("cossim", "pearson_corr"))
    args = parser.parse_args()
    params = vars(args)

    global_params, global_objects = config_roster(params)
    roster = DLCDRoster(global_params, global_objects)

    file_manager = global_objects["file_manager"]
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    save_dir = os.path.join(setting_dir, "que_smi_mat")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, f"{params['dataset_name']}_{params['similarity']}_{params['model_dir_name']}.npy")
    if not os.path.exists(save_path):
        Q_table = file_manager.get_q_table(params["dataset_name"])
        num_question = Q_table.shape[0]
        question_all = torch.tensor(list(range(num_question))).long().to(global_params["device"])
        model_dir_name = params["model_dir_name"]
        model = global_objects["models"][model_dir_name]
        q2c_transfer_table = global_objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = global_objects["dataset"]["q2c_mask_table"]
        if "NCD@@" in model_dir_name:
            question_emb = model.embed_layer.get_emb("question_diff", question_all).detach().cpu().numpy()
        elif "RCD@@" in model_dir_name:
            question_emb = model.embed_layer.get_emb("question", question_all).detach().cpu().numpy()
        else:
            raise NotImplementedError(f"model {model_dir_name} is not implemented the function of get all question emb")

        if params["similarity"] == "cossim":
            que_similarity = cosine_similarity_matrix(question_emb, axis=1)
        elif params["similarity"] == "pearson_corr":
            que_similarity = np.zeros((num_question, num_question))
            for i in range(num_question):
                for j in range(num_question):
                    si = question_emb[:, i]
                    sj = question_emb[:, j]
                    que_similarity[i][j] = pearson_similarity(si, sj)
        else:
            raise NotImplementedError(f'{params["similarity"]} is not implemented')
        
        np.save(save_path, que_similarity)