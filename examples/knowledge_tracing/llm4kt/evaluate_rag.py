import argparse
import os
import dspy
import inspect
import numpy as np

from config import FILE_MANAGER_ROOT
from predict import predict_rag

from edmine.utils.data_io import read_kt_file, read_json, read_csv, write_json
from edmine.utils.calculate import cosine_similarity_matrix
from edmine.data.FileManager import FileManager
from edmine.llm.dspy.remote_llm.GLM import GLM
from edmine.llm.dspy.remote_llm.BaiLian import BaiLian
from edmine.utils.parse import q2c_from_q_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="qwen-plus")
    parser.add_argument("--setting_name", type=str, default="pykt_setting")
    parser.add_argument("--dataset_name", type=str, default="moocradar-C746997")
    parser.add_argument("--emb_file_name", type=str, default="moocradar-C746997_qid2content_768.json")
    parser.add_argument("--test_file_name", type=str, default="moocradar-C746997-subtest-27.txt")
    parser.add_argument("--seq_start", type=int, default=171)
    parser.add_argument("--max_history", type=int, default=60, help="unit: day")
    parser.add_argument("--num2evaluate", type=int, default=3000)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--presence_penalty", type=float, default=1.8)
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(FILE_MANAGER_ROOT)
    processed_dir = file_manager.get_preprocessed_dir(params["dataset_name"])
    setting_dir = file_manager.get_setting_dir(params["setting_name"])

    q_table = file_manager.get_q_table(params["dataset_name"])
    q2c = q2c_from_q_table(q_table)
    kt_data = read_kt_file(os.path.join(setting_dir, params["test_file_name"]))
    question_id_map = read_csv(os.path.join(processed_dir, "question_id_map.csv"))
    concept_id_map = read_csv(os.path.join(processed_dir, "concept_id_map.csv"))
    question_id_map = question_id_map.to_dict()
    concept_id_map = concept_id_map.to_dict()
    question_meta = {
        q_id: {
            "text": q_text
        } for q_id, q_text in zip(question_id_map["mapped_id"].values(), question_id_map["text"].values())
    }
    concept_meta = {
        c_id: {
            "text": c_text
        } for c_id, c_text in zip(concept_id_map["mapped_id"].values(), concept_id_map["text"].values())
    }

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    question_emb_json = read_json(os.path.join(current_dir, "emb", params["emb_file_name"]))
    num_question = len(question_emb_json)
    question_emb = []
    for i in range(num_question):
        question_emb.append(np.array(question_emb_json[str(i)]))
    question_emb = np.stack(question_emb)
    que_sim_mat = cosine_similarity_matrix(question_emb, axis=1)
    similar_questions_np = np.argsort(-que_sim_mat, axis=1)[:, 1:num_question//100]
    similar_questions = {}
    for i in range(num_question):
        similar_questions[i] = similar_questions_np[i].tolist()

    # 选择LLM
    if "glm" in args.llm:
        dspy_lm = GLM(f"zhipu/{args.llm}")
    elif "qwen" in args.llm:
        dspy_lm = BaiLian(f"bailian/{args.llm}", top_p=params["top_p"], presence_penalty=params["presence_penalty"])
    else:
        raise NotImplementedError()
    dspy.configure(lm=dspy_lm)

    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(current_dir,
                               f"output/{args.llm}_rag_{args.test_file_name.replace('.txt', '')}_{args.seq_start}_{args.max_history}.json")
    prediction = predict_rag(dspy_lm, kt_data, question_meta, concept_meta, q2c, similar_questions, output_path, params)
    write_json(prediction, output_path)

