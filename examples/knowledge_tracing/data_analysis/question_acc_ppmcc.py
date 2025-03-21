import argparse
from collections import defaultdict

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_kt_file
from edmine.data.FileManager import FileManager
from edmine.utils.parse import q2c_from_q_table, get_ppmcc_no_error, cal_qc_acc4kt_data


# 分析数据集中学生做对知识点的正确率与习题正确率的关系
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
    
    # 统计习题正确率和出现次数
    corrects = defaultdict(int)
    counts = defaultdict(int)
    for item_data in kt_data:
        for q_id, correctness in zip(item_data["question_seq"], item_data["correctness_seq"]):
            corrects[q_id] += correctness
            counts[q_id] += 1
    all_ids = list(counts.keys())
    question_acc = {qc_id: corrects[qc_id] / float(counts[qc_id]) for qc_id in corrects}
    
    # correlation: key(int), 习题在数据集中出现次数，value(tuple[list, list]), 习题正确率和当前结果
    # 最多分析到历史练习知识点次数为50次
    correlation = {i+1: ([], []) for i in range(50)}
    for item_data in kt_data:
        history_count = defaultdict(int)
        history_correct = defaultdict(int)
        question_seq = item_data["question_seq"]
        correctness_seq = item_data["correctness_seq"]
        for q_id, correctness in zip(question_seq, correctness_seq):
            num_exercised = counts[q_id]
            if num_exercised > 0:
                correlation_key = min(50, num_exercised)
                correlation[correlation_key][0].append(question_acc[q_id])
                correlation[correlation_key][1].append(correctness)
            history_count[q_id] += 1
            history_correct[q_id] += correctness
                
    for k, v in correlation.items():
        num = len(v[0])
        ppmcc = float(get_ppmcc_no_error(v[0], v[1]))
        correlation[k] = (num, ppmcc)
        
    print(correlation)
    
    