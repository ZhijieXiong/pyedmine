import os
import argparse
import inspect
import numpy as np
from zhipuai import ZhipuAI
from tqdm import tqdm

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_csv, read_json, write_json
from edmine.data.FileManager import FileManager
from edmine.utils.calculate import cosine_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="glm-4-air")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--emb_file_name", type=str, default="xes3g5m_cid2content_256.json")
    parser.add_argument("--sim_th", type=float, default=0.75, help="相似度阈值")
    args = parser.parse_args()
    params = vars(args)

    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    concept_emb_json = read_json(os.path.join(current_dir, "../emb", params["emb_file_name"]))
    num_concept = len(concept_emb_json)
    concept_embs = []
    for i in range(num_concept):
        concept_embs.append(np.array(concept_emb_json[str(i)]))

    file_manager = FileManager(FILE_MANAGER_ROOT)
    processed_dir = file_manager.get_preprocessed_dir(params["dataset_name"])
    dataset_name = params["dataset_name"]
    dimensions = int(params["emb_file_name"].split("_")[-1].replace(".json", ""))
    concept_id_map = read_csv(os.path.join(processed_dir, "concept_id_map.csv")).to_dict()
    concept_meta = {
        concept_id: {
            "text": concept_text
        } for concept_id, concept_text in zip(concept_id_map["mapped_id"].values(), concept_id_map["text"].values())
    }
    concepts = []
    for i in range(len(concept_meta)):
        concepts.append(concept_meta[i]["text"])

    prerequisites_path = os.path.join(current_dir, "../output",
                                      f"concept_directed_graph_{dataset_name}_{params['llm_name']}_{dimensions}_{params['sim_th']}.json")
    prerequisites = {}
    if os.path.exists(prerequisites_path):
        pre = read_json(prerequisites_path)
        for k, v in pre.items():
            prerequisites[int(k)] = v
    try:
        client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        for c_id, concept in enumerate(tqdm(concepts)):
            if c_id in prerequisites:
                continue
            response = client.chat.completions.create(
                model=params["llm_name"],
                messages=[
                    {
                        "role": "user",
                        "content": "请你告诉我距离输入知识点层次最近的先修知识点，并按照示例的格式输出（多个知识点用`、`分隔）\n"
                                   "注意：若输入知识点为中文，则输出也为中文，同理若输入知识点为英文，则输出知识点也为英文\n"
                                   "示例1\n输入：分数运算\n输出：通分、约分\n"
                                   "示例2\n输入：通分\n输出：分数的基本概念和性质、最小公倍数\n"
                                   "示例3\n输入：10以内的加减法\n输出：根知识点\n"
                                   "示例4\n输入：Fraction operations\n输出：Common denominator、reduction\n"
                                   "示例5\n输入：Addition and subtraction within 10\n输出：Root concept\n"
                                   f"输入：{concept}\n输出："
                    }
                ],
                temperature=0.1,
                max_tokens=50,
            )
            r_text = response.choices[0].message.content.strip()
            if "根知识点" in r_text or "Root concept" in r_text:
                continue
            r_concepts = list(filter(lambda t: len(t) > 0, list(map(lambda t: t.strip(), r_text.split("、")))))
            if len(r_concepts) == 0:
                continue
            response = client.embeddings.create(
                model="embedding-3",
                input=r_concepts,
                dimensions=dimensions
            )
            target_embs = []
            for emb_data in response.data:
                target_embs.append(np.array(emb_data.embedding))
            sim_mat = cosine_similarity(target_embs, concept_embs)
            most_sim_c_ids = np.argsort(-sim_mat, axis=1)[:, 0].tolist()
            for i, sim_c_id in enumerate(most_sim_c_ids):
                if sim_mat[i, sim_c_id] > params["sim_th"]:
                    if c_id not in prerequisites:
                        prerequisites[c_id] = [sim_c_id]
                    else:
                        prerequisites[c_id].append(sim_c_id)
            if c_id in prerequisites:
                prerequisites[c_id] = list(set(prerequisites[c_id]))
    except:
        write_json(prerequisites, prerequisites_path)
    write_json(prerequisites, prerequisites_path)

