import os
import inspect
from sklearn.metrics import accuracy_score

from edmine.utils.data_io import read_json


current_file_name = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(current_file_name)

outputs = os.listdir(os.path.join(current_dir, "output"))
for output in outputs:
    prediction = read_json(os.path.join(current_dir, "output", output))
    pls = []
    gts = []
    for p in prediction.values():
        pls.append(p["pl"])
        gts.append(p["gt"])
    print(f"{output.replace('.json', '')}: {accuracy_score(gts, pls)}")
