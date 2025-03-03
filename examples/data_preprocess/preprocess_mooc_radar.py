import json
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from edmine.utils.data_io import read_json, write_kt_file, write_json
from edmine.utils.parse import q2c_from_q_table, get_kt_data_statics


def time_str2timestamp(time_str):
    return int(time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    data_path = "/Users/dream/myProjects/pyedmine/dataset/dataset_raw/moocradar/student-problem-coarse.json"
    problem_data_path = "/Users/dream/myProjects/pyedmine/dataset/dataset_raw/moocradar/problem.json"
    data_process_dir = "/Users/dream/myProjects/pyedmine/dataset/dataset_preprocessed"
    # 最大3个课程的数据集C_2287011（政治），C_797404（酒）, C_746997（模电）
    target_course_id = "C_746997"

    problem_data = []
    with open(problem_data_path, "r") as f:
        for line in f:
            problem_data.append(json.loads(line))

    num_question = len(problem_data)
    q_id_map = {}
    concepts = set()
    courses = set()
    courses_example = {}
    question_meta_data = {}
    # 只有2个错误
    num_err_eval = 0
    for i, p in enumerate(problem_data):
        q_id_map[p["problem_id"]] = i
        concepts.update(p["concepts"])
        courses.add(p["course_id"])
        if p["course_id"] not in courses_example:
            courses_example[p["course_id"]] = i

        try:
            p_detail = eval(p["detail"])
            question_meta_data[i] = {
                "original_id": p["problem_id"],
                "course_id": p["course_id"],
                "type": p_detail["typetext"],
                "content": p_detail["content"],
                "option": p_detail["option"],
                "answer": p_detail["answer"],
                "concept_ids": p["concepts"],
            }
        except:
            print(p["problem_id"])
            num_err_eval += 1
            question_meta_data[i] = {
                "original_id": p["problem_id"],
                "course_id": p["course_id"],
                "concept_ids": p["concepts"]
            }

    # 手动处理错误eval
    question_meta_data[q_id_map["Pm_1308411"]].update({
        "type": "判断题",
        "content": "政府购买与政府支出没有区别",
        "option": None,
        "answer": ["false"],
    })
    question_meta_data[q_id_map["Pm_8045638"]].update({
        "type": "单选题",
        "content": "以下哪一项不属于创业者必须具备的能力:()",
        "option": {'A': '人际协调能力', 'B': '开拓创新能力', 'C': '组织管理能力', 'D': '自我执行能力'},
        "answer": ["D"],
    })

    # 看一下有哪些课程
    print(f"There are {len(courses)} courses")
    for course_id in courses_example:
        print(course_id)
        print(question_meta_data[courses_example[course_id]]["content"])

    num_concept = len(concepts)
    for q_id in question_meta_data:
        question_meta_data[q_id]["concept_ids"] = (
            list(map(lambda c_name: c_name.strip(), question_meta_data[q_id]["concept_ids"]))
        )

    data = read_json(data_path)
    data_target_course = []
    course_count = defaultdict(int)
    # 预处理数据，丢弃不完整的数据记录，并计算每个课程的数据量
    for user_data in data:
        seq = user_data["seq"]
        for item in seq:
            item["submit_time"] = time_str2timestamp(item["submit_time"])
        # 按时间排序
        seq.sort(key=lambda x: x["submit_time"])

        item_data = {"user_id": seq[0]["user_id"]}
        question_seq = []
        correctness_seq = []
        time_seq = []
        num_attempt_seq = []
        for item in seq:
            q_id = item["problem_id"]
            is_correct = item["is_correct"]
            submit_time = item["submit_time"]
            attempts = item["attempts"]
            if is_correct in [0, 1] and attempts > 0:
                course_id = item["course_id"]
                course_count[course_id] += 1
                if course_id == target_course_id:
                    question_seq.append(q_id)
                    correctness_seq.append(is_correct)
                    time_seq.append(submit_time)
                    num_attempt_seq.append(attempts)
        item_data["question_seq"] = question_seq
        item_data["correctness_seq"] = correctness_seq
        item_data["time_seq"] = time_seq
        item_data["num_attempt_seq"] = num_attempt_seq
        seq_len = len(item_data["question_seq"])
        if seq_len > 1:
            item_data["seq_len"] = seq_len
            data_target_course.append(item_data)

    course_count = [(course_id, n) for course_id, n in course_count.items()]
    course_count.sort(key=lambda x: x[1], reverse=True)
    print(f"example of {course_count[0][0]}: {question_meta_data[courses_example[course_count[0][0]]]['content']}")
    print(f"example of {course_count[1][0]}: {question_meta_data[courses_example[course_count[1][0]]]['content']}")
    print(f"example of {course_count[2][0]}: {question_meta_data[courses_example[course_count[2][0]]]['content']}")

    # 判断该课程下的习题是否有多知识点
    course_q_meta_data = {}
    for q_meta in question_meta_data.values():
        if q_meta["course_id"] == target_course_id:
            course_q_meta_data[q_meta["original_id"]] = q_meta

    # 重映射所选数据集的习题id和知识点id，以及Q table
    course_question_ids = list(course_q_meta_data.keys())
    course_concept_ids = []
    for q in course_q_meta_data.values():
        course_concept_ids += q["concept_ids"]
    course_concept_ids = list(set(course_concept_ids))
    course_q_map = {original_id: new_id for new_id, original_id in enumerate(course_question_ids)}
    course_c_map = {original_id: new_id for new_id, original_id in enumerate(course_concept_ids)}
    Q_table = np.zeros((len(course_question_ids), len(course_concept_ids)), dtype=int)
    course_q_meta_data_final = {}
    for q_meta in course_q_meta_data.values():
        q_id = course_q_map[q_meta["original_id"]]
        correspond_c = list(map(lambda c_id: course_c_map[c_id], q_meta["concept_ids"]))
        q_meta["concepts"] = q_meta["concept_ids"]
        q_meta["concept_ids"] = correspond_c
        course_q_meta_data_final[q_id] = q_meta
        Q_table[[q_id] * len(correspond_c), correspond_c] = [1] * len(correspond_c)
    question2concept = q2c_from_q_table(Q_table)

    # 将data数据中的question seq重映射为int id，如果has_multi_concept为False的话，添加concept_seq
    for i, item_data in enumerate(data_target_course):
        item_data["user_id"] = i
        item_data["question_seq"] = [course_q_map[q_ori_id] for q_ori_id in item_data["question_seq"]]

    data_dir = os.path.join(data_process_dir, f"moocradar-{target_course_id}")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    num_interaction = 0
    for item_data in data_target_course:
        num_interaction += item_data["seq_len"]

    data_statics = get_kt_data_statics(data_target_course, Q_table)
    data_statics_preprocessed = {
        "num_interaction": num_interaction,
        "num_user": len(data_target_course),
        "num_concept": len(course_concept_ids),
        "num_question": len(course_question_ids),
        "average_seq_len": data_statics["ave_seq_len"],
        "average_question_acc": data_statics["ave_que_acc"],
        "question_sparsity": data_statics["que_sparsity"],
        "concept_sparsity": data_statics["concept_sparsity"]
    }
    question_id_map = {
        "mapped_id": [],
        "original_id": []
    }
    for question_id, question_mapped_id in course_q_map.items():
        question_id_map["mapped_id"].append(question_mapped_id)
        question_id_map["original_id"].append(question_id)
    question_id_map = pd.DataFrame(question_id_map)
    concept_id_map = {
        "mapped_id": [],
        "original_id": []
    }
    for concept_id, concept_mapped_id in course_c_map.items():
        concept_id_map["mapped_id"].append(concept_mapped_id)
        concept_id_map["original_id"].append(concept_id)
    concept_id_map = pd.DataFrame(concept_id_map)

    # 必须有的数据
    write_kt_file(data_target_course, os.path.join(data_dir, "data.txt"))
    np.save(os.path.join(data_dir, "Q_table.npy"), Q_table)
    # 非必需
    write_json(data_statics_preprocessed, os.path.join(data_dir, "statics_preprocessed.json"))
    concept_id_map.to_csv(os.path.join(data_dir, "concept_id_map.csv"), index=False)
    question_id_map.to_csv(os.path.join(data_dir, "question_id_map.csv"), index=False)
    # write_json(course_q_meta_data_final, os.path.join(data_dir, "question_meta.json"))



