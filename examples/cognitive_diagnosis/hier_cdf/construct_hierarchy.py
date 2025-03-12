import argparse
import os
import torch
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BicScore, HillClimbSearch

from utils import load_model, FILE_MANAGER_ROOT

from edmine.data.FileManager import FileManager


MODEL_DIR_NAMES = [
    "NCD@@ncd_setting@@assist2009_train_fold_0@@seed_0@@2025-03-11@01-46-58",
    "NCD@@ncd_setting@@assist2009_train_fold_1@@seed_0@@2025-03-11@01-55-07",
    "NCD@@ncd_setting@@assist2009_train_fold_2@@seed_0@@2025-03-11@02-03-19",
    "NCD@@ncd_setting@@assist2009_train_fold_3@@seed_0@@2025-03-11@02-11-40",
    "NCD@@ncd_setting@@assist2009_train_fold_3@@seed_0@@2025-03-11@02-11-40"
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据配置
    parser.add_argument("--setting_name", type=str, default="ncd_setting")
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    # 加载CD模型
    parser.add_argument("--model_name", type=str, help="文件名", default="saved.ckt")
    parser.add_argument("--model_name_in_ckt", type=str, help="文件名", default="best_valid")
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(FILE_MANAGER_ROOT)
    setting_dir = file_manager.get_setting_dir(params["setting_name"])
    save_dir = os.path.join(setting_dir, "HierCDF")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, f"{params['dataset_name']}_hierarchy.csv")
    if not os.path.exists(save_path):
        Q_table = file_manager.get_q_table(params["dataset_name"])
        num_concept = Q_table.shape[1]
        data_statics_path = os.path.join(setting_dir, f"{params['dataset_name']}_statics.txt")
        with open(data_statics_path, "r") as f:
            s = f.readline()
            num_user = int(s.split(":")[1].strip())
            
        user_embs = np.zeros((num_user, num_concept))
        for model_dir_name in MODEL_DIR_NAMES:
            params["model_dir_name"] = model_dir_name
            model = load_model(params)
            user_all = torch.tensor(list(range(num_user))).long().to(next(model.parameters()).device)
            user_embs += model.get_knowledge_state(user_all).detach().cpu().numpy()
        
        user_emb = user_embs / len(MODEL_DIR_NAMES)
        edges = []
        for j in range(num_concept):
            for k in range(num_concept):
                if j == k:
                    continue
                # 计算条件概率
                p_j1 = user_emb[:, j].mean()
                p_k1_given_j1 = user_emb[user_emb[:, j] > 0.5, k].mean()
                p_k1_given_j0 = user_emb[user_emb[:, j] <= 0.5, k].mean()
                if p_k1_given_j1 > p_k1_given_j0 + 0.01:  # 阈值可调
                    edges.append((j, k))
        
        print(edges)
        # data = pd.DataFrame(user_emb, columns=[f"K{i}" for i in range(num_concept)])
        # scoring_method = BicScore(data)
        # hc = HillClimbSearch(data)
        # bn = BayesianNetwork(hc.estimate(scoring_method, white_list=edges).edges())
        # bn.fit(data, estimator=MaximumLikelihoodEstimator)
        # hierarchy_df = pd.DataFrame(bn.edges(), columns=["from", "to"])
        # hierarchy_df.to_csv(save_path, index=False)