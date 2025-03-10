import argparse
from collections import defaultdict

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table, get_ppmcc_no_error


# 分析数据集中学生做对知识点的正确率与历史练习的关系
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    args = parser.parse_args()
    params = vars(args)
    
    file_manager = FileManager(FILE_MANAGER_ROOT)
    data_processed_dir = file_manager.get_preprocessed_dir(params["dataset_name"])
    data_path = file_manager.get_preprocessed_path(params["dataset_name"])
    kt_data = read_kt_file(data_path)
    q_table = file_manager.get_q_table(params['dataset_name'])
    q2c = q2c_from_q_table(q_table)
    
    # acc_dict: key(int), 历史练习知识点次数，value(tuple[list, list]), 历史正确率和当前结果
    # 最多分析到历史练习知识点次数为20次
    correlation = {i+1: ([], []) for i in range(20)}
    for item_data in kt_data:
        history_count = defaultdict(int)
        history_correct = defaultdict(int)
        question_seq = item_data["question_seq"]
        correctness_seq = item_data["correctness_seq"]
        for q_id, correctness in zip(question_seq, correctness_seq):
            c_ids = q2c[q_id]
            for c_id in c_ids:
                num_exercised = history_count[c_id]
                num_correct = history_correct[c_id]
                if num_exercised > 0:
                    correlation_key = min(20, num_exercised)
                    correlation[correlation_key][0].append(num_correct / num_exercised)
                    correlation[correlation_key][1].append(correctness)
                history_count[c_id] += 1
                history_correct[c_id] += correctness
                
    for k, v in correlation.items():
        num = len(v[0])
        ppmcc = float(get_ppmcc_no_error(v[0], v[1]))
        correlation[k] = (num, ppmcc)
        
    print(correlation)
    
    