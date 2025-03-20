import os
import inspect
import argparse
from zhipuai import ZhipuAI

from config import FILE_MANAGER_ROOT

from edmine.utils.data_io import read_csv, write_json
from edmine.data.FileManager import FileManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="moocradar-C746997")
    parser.add_argument("--target_item", type=str, default="question", choices=("concept", "question"))
    parser.add_argument("--dimensions", type=int, default=768)
    args = parser.parse_args()
    params = vars(args)

    file_manager = FileManager(FILE_MANAGER_ROOT)
    processed_dir = file_manager.get_preprocessed_dir(params["dataset_name"])
    target_item = params["target_item"]
    dataset_name = params["dataset_name"]
    dimensions = params["dimensions"]
    item_id_map = read_csv(
        os.path.join(processed_dir, "concept_id_map.csv" if target_item == "concept" else "question_id_map.csv")
    ).to_dict()
    item_meta = {
        item_id: {
            "text": item_text
        } for item_id, item_text in zip(item_id_map["mapped_id"].values(), item_id_map["text"].values())
    }
    items = []
    for i in range(len(item_meta)):
        items.append(item_meta[i]["text"])
    num_item = len(items)
    item_embs = {}
    for i in range(0, num_item, 64):
        client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        response = client.embeddings.create(
            model="embedding-3",
            input=items[i:i+64],
            dimensions=dimensions
        )
        for j, emb_data in enumerate(response.data):
            item_embs[i + j] = emb_data.embedding

    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    save_path = os.path.join(
        current_dir,
        "..",
        "emb",
        f"{dataset_name}_{'cid' if target_item == 'concept' else 'qid'}2content_{dimensions}.json"
    )
    write_json(item_embs, save_path)
