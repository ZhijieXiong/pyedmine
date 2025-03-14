import os
import inspect
from sklearn.metrics import accuracy_score

from edmine.utils.data_io import read_json


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)

outputs = os.listdir(os.path.join(current_dir, "output"))
outputs = sorted(outputs, key=lambda x: int(x.split("_")[-1].replace(".json", "")))
target_model = "long"
for output in outputs:
    if target_model in output:
        prediction = read_json(os.path.join(current_dir, "output", output))
        pls = []
        gts = []
        i_tokens = 0
        o_tokens = 0
        for p in prediction.values():
            pls.append(p["pl"])
            gts.append(p["gt"])
            i_tokens += p.get("i_tokens", 0)
            o_tokens += p.get("o_tokens", 0)
        print(f"{output.replace('.json', '')}, num: {len(prediction)}, acc: {accuracy_score(gts, pls)}"
              f"{'' if i_tokens == 0 else f', input tokens: {i_tokens//1000}k, output tokens: {o_tokens//1000}k, total tokens: {(i_tokens+o_tokens)//1000}k'}")
