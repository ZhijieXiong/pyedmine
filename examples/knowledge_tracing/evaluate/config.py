import json
import os
import inspect
from collections import defaultdict

from utils import *

from edmine.data.FileManager import FileManager
from edmine.config.data import config_q_table
from edmine.config.basic import config_logger
from edmine.config.model import config_general_dl_model
from edmine.model.load_model import load_dl_model
from edmine.utils.check import check_kt_seq_start
from edmine.utils.log import get_now_time
from edmine.utils.data_io import read_kt_file, read_json, write_json


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)
settings_path = os.path.join(current_dir, "../../settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)
FILE_MANAGER_ROOT = settings["FILE_MANAGER_ROOT"]
MODEL_DIR = settings["MODELS_DIR"]


def Laplace(num_count, num_correct):
    return float((num_correct + 1) / (num_count + 2))


def get_concept_acc(train_data, q2c, num_concept=None):
    concept_count = {}
    concept_correct = {}
    for user_data in train_data:
        seq_len = user_data["seq_len"]
        question_seq = user_data["question_seq"][:seq_len]
        correctness_seq = user_data["correctness_seq"][:seq_len]
        for q, correctness in zip(question_seq, correctness_seq):
            for c in q2c[q]:
                if c not in concept_count:
                    concept_count[c] = 0
                if c not in concept_correct:
                    concept_correct[c] = 0
                concept_count[c] += 1
                concept_correct[c] += correctness
    pi = float(sum(concept_correct.values()) / sum(concept_count.values()))
    if num_concept is None:
        concepts = concept_count.keys()
    else:
        concepts = list(range(num_concept))
    concept_acc = {}
    for c in concepts:
        if c not in concept_count:
            concept_acc[c] = pi
        else:
            concept_acc[c] = Laplace(concept_count[c], concept_correct[c])
    return concept_acc, concept_count, concept_correct


def get_question_acc(train_data, num_question=None):
    question_count = {}
    question_correct = {}
    for user_data in train_data:
        seq_len = user_data["seq_len"]
        question_seq = user_data["question_seq"][:seq_len]
        correctness_seq = user_data["correctness_seq"][:seq_len]
        for q, correctness in zip(question_seq, correctness_seq):
            if q not in question_count:
                question_count[q] = 0
            if q not in question_correct:
                question_correct[q] = 0
            question_count[q] += 1
            question_correct[q] += correctness
    pi = float(sum(question_correct.values()) / sum(question_count.values()))
    if num_question is None:
        questions = question_count.keys()
    else:
        questions = list(range(num_question))
    question_acc = {}
    for q in questions:
        if q not in question_count:
            question_acc[q] = pi
        else:
            question_acc[q] = Laplace(question_count[q], question_correct[q])
    return question_acc, question_count, question_correct


def config_sequential_dlkt(local_params):
    global_params = {}
    global_objects = {"file_manager": FileManager(FILE_MANAGER_ROOT)}
    if local_params.get("save_log", False):
        log_path = os.path.join(MODEL_DIR, local_params["model_dir_name"],
                                f"evaluate_log@{get_now_time().replace(' ', '@').replace(':', '-')}.txt")
    else:
        log_path = None
    if local_params.get("save_all_sample", False):
        all_sample_path = os.path.join(MODEL_DIR, local_params["model_dir_name"],
                                       f"all_sample_evaluation.txt")
    else:
        all_sample_path = None
    global_params["all_sample_path"] = all_sample_path
    config_logger(local_params, global_objects, log_path)
    config_general_dl_model(local_params, global_params)
    check_kt_seq_start(local_params.get("seq_start", 2))
    global_params["sequential_dlkt"] = {
        "seq_start": local_params.get("seq_start", 2),
        "que_start": local_params.get("que_start", 0),
        "question_cold_start": local_params.get("question_cold_start", -1),
        "user_cold_start": local_params.get("user_cold_start", 0),
        "multi_step_accumulate": local_params.get("multi_step_accumulate", False),
        "multi_step_overall": local_params.get("multi_step_overall", False),
        "multi_step": local_params.get("multi_step", 1),
        "use_core": local_params.get("use_core", False),
        "evaluate_overall": local_params.get("evaluate_overall", True),
        "use_bes": local_params.get("use_bes", False),
        "user_hard_th": local_params.get("user_hard_th", 0),
        "concept_hard_th": local_params.get("concept_hard_th", 0),
        "question_hard_th": local_params.get("question_hard_th", 0),
    }
    config_q_table(local_params, global_params, global_objects)
    
    model_name, setting_name, train_file_name = get_model_info(local_params["model_dir_name"])
    setting_dir = global_objects["file_manager"].get_setting_dir(setting_name)
    
    cold_start_dir = os.path.join(setting_dir, "data4cold_start")
    if not os.path.exists(cold_start_dir):
        os.mkdir(cold_start_dir)
    question_cold_start = global_params["sequential_dlkt"]["question_cold_start"]
    if question_cold_start >= 0:
        cold_start_question_path = os.path.join(cold_start_dir,
                                                f"{train_file_name}_cold_start_question_{question_cold_start}.json")
        if os.path.exists(cold_start_question_path):
            global_objects["cold_start_question"] = read_json(cold_start_question_path)
        else:
            train_file_path = os.path.join(setting_dir, train_file_name + ".txt")
            train_data = read_kt_file(train_file_path)
            num_q_in_train = defaultdict(int)
            for item_data in train_data:
                seq_len = item_data["seq_len"]
                question_seq = item_data["question_seq"][:seq_len]
                for question_id in question_seq:
                    num_q_in_train[question_id] += 1
            global_objects["num_q_in_train"] = num_q_in_train
            global_objects["cold_start_question"] = []
            for question_id, num_question in num_q_in_train.items():
                if num_question <= question_cold_start:
                    global_objects["cold_start_question"].append(question_id)
            write_json(global_objects["cold_start_question"], cold_start_question_path)
            
    warm_start_dir = os.path.join(setting_dir, "data4warm_start")
    if not os.path.exists(warm_start_dir):
        os.mkdir(warm_start_dir)
    que_start = global_params["sequential_dlkt"]["que_start"]
    if que_start > 0:
        warm_start_question_path = os.path.join(warm_start_dir,
                                                f"{train_file_name}_warm_start_question_{que_start}.json")
        if os.path.exists(warm_start_question_path):
            global_objects["warm_start_question"] = read_json(warm_start_question_path)
        else:
            train_file_path = os.path.join(setting_dir, train_file_name + ".txt")
            train_data = read_kt_file(train_file_path)
            num_q_in_train = defaultdict(int)
            for item_data in train_data:
                seq_len = item_data["seq_len"]
                question_seq = item_data["question_seq"][:seq_len]
                for question_id in question_seq:
                    num_q_in_train[question_id] += 1
            global_objects["num_q_in_train"] = num_q_in_train
            global_objects["warm_start_question"] = []
            for question_id, num_question in num_q_in_train.items():
                if num_question >= que_start:
                    global_objects["warm_start_question"].append(question_id)
            write_json(global_objects["warm_start_question"], warm_start_question_path)
    
    dataset_name = local_params["dataset_name"]
    bes_dir = os.path.join(setting_dir, "data4bes")
    if not os.path.exists(bes_dir):
        os.mkdir(bes_dir)
    use_bes = global_params["sequential_dlkt"]["use_bes"]
    if use_bes:
        train_file_path = os.path.join(setting_dir, train_file_name + ".txt")
        train_data = read_kt_file(train_file_path)
        
        # 统计知识点信息
        # 计算知识点的样本参照r：（1）利用经过Laplace平滑的acc作为样本参照；（2）用 Naive-Bayes 风格的似然比 合成样本参照
        if "single-concept" in dataset_name:
            concept_acc_path = os.path.join(bes_dir, f"{train_file_name}_concept_acc_single_concept.json")
            concept_lr_path = os.path.join(bes_dir, f"{train_file_name}_concept_likelihood_ratio_single_concept.json")
        else:
            concept_acc_path = os.path.join(bes_dir, f"{train_file_name}_concept_acc.json")
            concept_lr_path = os.path.join(bes_dir, f"{train_file_name}_concept_likelihood_ratio.json")
        num_concept = global_objects["dataset"]["q_table"].shape[1]
        if not os.path.exists(concept_acc_path) or not os.path.exists(concept_lr_path):
            q2c = global_objects["dataset"]["q2c"]
            concept_acc, concept_count, concept_correct = get_concept_acc(train_data, q2c, num_concept)
            pi = float(sum(concept_correct.values()) / sum(concept_count.values()))
            global_objects["concept_acc"] = concept_acc
            write_json(concept_acc, concept_acc_path)
            
            concept_lr = {}
            n_pos = sum(concept_correct.values())
            n_neg = sum(concept_count.values()) - n_pos
            for c in range(num_concept):
                if c not in concept_count:
                    # 等价于忽略该因子
                    concept_lr[c] = 1
                else:
                    n_concept = concept_count[c]
                    n_c_pos = concept_correct[c]
                    n_c_neg = n_concept - n_c_pos
                    P_c_pos = (n_c_pos + 1) / (n_pos + n_concept)
                    P_c_neg = (n_c_neg + 1) / (n_neg + n_concept)
                    concept_lr[c] = float(P_c_pos / P_c_neg)
            global_objects["concept_lr"] = concept_lr
            concept_lr["pi"] = pi
            write_json(concept_lr, concept_lr_path)
        else:
            concept_acc = read_json(concept_acc_path)
            concept_lr = read_json(concept_lr_path)
            global_objects["concept_acc"] = {int(c): acc for c, acc in concept_acc.items()}
            global_objects["concept_lr"] = {int(c) if c != "pi" else c: lr for c, lr in concept_lr.items()}
        
        # 统计习题信息：同concept
        question_acc_path = os.path.join(bes_dir, f"{train_file_name}_question_acc.json")
        question_lr_path = os.path.join(bes_dir, f"{train_file_name}_question_likelihood_ratio.json")
        num_question = global_objects["dataset"]["q_table"].shape[0]
        if not os.path.exists(question_acc_path) or not os.path.exists(question_lr_path):
            question_acc, question_count, question_correct = get_question_acc(train_data, num_question)
            pi = float(sum(question_correct.values()) / sum(question_count.values()))
            global_objects["question_acc"] = question_acc
            write_json(question_acc, question_acc_path)
            
            question_lr = {}
            n_pos = sum(question_correct.values())
            n_neg = sum(question_count.values()) - n_pos
            for q in range(num_question):
                if q not in question_count:
                    # 等价于忽略该因子
                    question_lr[q] = 1
                else:
                    n_question = question_count[q]
                    n_q_pos = question_correct[q]
                    n_q_neg = n_question - n_q_pos
                    P_q_pos = (n_q_pos + 1) / (n_pos + n_question)
                    P_q_neg = (n_q_neg + 1) / (n_neg + n_question)
                    question_lr[q] = float(P_q_pos / P_q_neg)
            global_objects["question_lr"] = question_lr
            question_lr["pi"] = pi
            write_json(question_lr, question_lr_path)
        else:
            question_acc = read_json(question_acc_path)
            question_lr = read_json(question_lr_path)
            global_objects["question_acc"] = {int(q): acc for q, acc in question_acc.items()}
            global_objects["question_lr"] = {int(q) if q != "pi" else q: lr for q, lr in question_lr.items()}
            
    hard_sample_dir = os.path.join(setting_dir, "data4hard_sample")
    if not os.path.exists(hard_sample_dir):
        os.mkdir(hard_sample_dir)
        
    concept_hard_th = global_params["sequential_dlkt"]["concept_hard_th"]
    if concept_hard_th > 0:
        if "single-concept" in dataset_name:
            concept_acc_path = os.path.join(hard_sample_dir, f"{train_file_name}_concept_acc_single_concept.json")
        else:
            concept_acc_path = os.path.join(hard_sample_dir, f"{train_file_name}_concept_acc.json")
        if not os.path.exists(concept_acc_path):
            train_file_path = os.path.join(setting_dir, train_file_name + ".txt")
            train_data = read_kt_file(train_file_path)
            q2c = global_objects["dataset"]["q2c"]
            concept_acc, concept_count, concept_correct = get_concept_acc(train_data, q2c)
            global_objects["concept_acc4hard_sample"] = concept_acc
            write_json(concept_acc, concept_acc_path)
        else:
            concept_acc = read_json(concept_acc_path)
            global_objects["concept_acc4hard_sample"] = {int(c): acc for c, acc in concept_acc.items()}
            
    question_hard_th = global_params["sequential_dlkt"]["question_hard_th"]
    if question_hard_th > 0:
        question_acc_path = os.path.join(hard_sample_dir, f"{train_file_name}_question_acc.json")
        if not os.path.exists(question_acc_path):
            train_file_path = os.path.join(setting_dir, train_file_name + ".txt")
            train_data = read_kt_file(train_file_path)
            question_acc, question_count, question_correct = get_question_acc(train_data)
            global_objects["question_acc4hard_sample"] = question_acc
            write_json(question_acc, question_acc_path)
        else:
            question_acc = read_json(question_acc_path)
            global_objects["question_acc4hard_sample"] = {int(q): acc for q, acc in question_acc.items()}
    
    # ABQR的config必须放在load_dl_model前面，因为初始化模型是需要gcn_adj
    if model_name == "ABQR":
        config_abqr(local_params, global_params, global_objects, setting_name)
        
    model_dir = os.path.join(MODEL_DIR, local_params["model_dir_name"])
    model = load_dl_model(global_params, global_objects,
                          model_dir, local_params["model_file_name"], local_params["model_name_in_ckt"])
    
    if model_name == "DIMKT":
        config_dimkt(local_params, global_params, global_objects, setting_name, train_file_name)
    if model_name == "LPKT":
        config_lpkt(global_params, global_objects)
    if model_name == "HDLPKT":
        config_hdlpkt(global_params, global_objects)
    if model_name == "LBKT":
        config_lbkt(global_params, global_objects)
    if model_name == "QDCKT":
        config_qdckt(global_params, global_objects, setting_name, train_file_name)
    if model_name == "GRKT":
        config_grkt(global_objects, global_objects, setting_name, train_file_name)
    
    global_params["evaluator_config"] = {"model_name": model_name}
    global_objects["models"] = {model_name: model}

    return global_params, global_objects
