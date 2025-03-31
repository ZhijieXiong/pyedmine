import os
import torch

from edmine.utils.parse import str_dict2params
from edmine.utils.data_io import read_json
# KT
from edmine.model.sequential_kt_model.DKT import DKT
from edmine.model.sequential_kt_model.qDKT import qDKT
from edmine.model.sequential_kt_model.DKVMN import DKVMN
from edmine.model.sequential_kt_model.ATKT import ATKT
from edmine.model.sequential_kt_model.SimpleKT import SimpleKT
from edmine.model.sequential_kt_model.AKT import AKT
from edmine.model.sequential_kt_model.DIMKT import DIMKT
from edmine.model.sequential_kt_model.LPKT import LPKT
from edmine.model.sequential_kt_model.LBKT import LBKT
from edmine.model.sequential_kt_model.QIKT import QIKT
from edmine.model.sequential_kt_model.SparseKT import SparseKT
from edmine.model.sequential_kt_model.QDCKT import QDCKT
from edmine.model.sequential_kt_model.SKVMN import SKVMN
from edmine.model.sequential_kt_model.DKTForget import DKTForget
from edmine.model.sequential_kt_model.MIKT import MIKT
from edmine.model.sequential_kt_model.ATDKT import ATDKT
from edmine.model.sequential_kt_model.CLKT import CLKT
from edmine.model.sequential_kt_model.DTransformer import DTransformer
from edmine.model.sequential_kt_model.GRKT import GRKT
from edmine.model.sequential_kt_model.CKT import CKT
from edmine.model.sequential_kt_model.HDLPKT import HDLPKT
# CD
from edmine.model.cognitive_diagnosis_model.NCD import NCD
from edmine.model.cognitive_diagnosis_model.IRT import IRT
from edmine.model.cognitive_diagnosis_model.MIRT import MIRT
from edmine.model.cognitive_diagnosis_model.DINA import DINA
from edmine.model.cognitive_diagnosis_model.RCD import RCD
from edmine.model.cognitive_diagnosis_model.HyperCD import HyperCD
from edmine.model.cognitive_diagnosis_model.HierCDF import HierCDF
# ER
from edmine.model.exercise_recommendation_model.KG4EX import KG4EX
from edmine.model.sequential_kt_model.DKT_KG4EX import DKT_KG4EX


model_table = {
    "DKT": DKT,
    "DKT_KG4EX": DKT_KG4EX,
    "qDKT": qDKT,
    "DKVMN": DKVMN,
    "ATKT": ATKT,
    "SimpleKT": SimpleKT,
    "AKT": AKT,
    "DIMKT": DIMKT,
    "LPKT": LPKT,
    "LBKT": LBKT,
    "SparseKT": SparseKT,
    "QIKT": QIKT,
    "QDCKT": QDCKT,
    "SKVMN": SKVMN,
    "DKTForget": DKTForget,
    "MIKT": MIKT,
    "ATDKT": ATDKT,
    "CLKT": CLKT,
    "DTransformer": DTransformer,
    "GRKT": GRKT,
    "CKT": CKT,
    "HDLPKT": HDLPKT,
    "NCD": NCD,
    "IRT": IRT,
    "MIRT": MIRT,
    "DINA": DINA,
    "RCD": RCD,
    "HyperCD": HyperCD,
    "HierCDF": HierCDF,
    "KG4EX": KG4EX
}


def load_dl_model(global_params, global_objects, save_model_dir, ckt_name="saved.ckt", model_name_in_ckt="best_valid"):
    params_path = os.path.join(save_model_dir, "params.json")
    saved_params = read_json(params_path)
    global_params["models_config"] = str_dict2params(saved_params["models_config"])

    ckt_path = os.path.join(save_model_dir, ckt_name)
    model_name = os.path.basename(save_model_dir).split("@@")[0]
    model_class = model_table[model_name]
    model = model_class(global_params, global_objects).to(global_params["device"])
    if global_params["device"] == "cpu":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('cpu'), weights_only=True)
    elif global_params["device"] == "mps":
        saved_ckt = torch.load(ckt_path, map_location=torch.device('mps'), weights_only=True)
    else:
        saved_ckt = torch.load(ckt_path, weights_only=True)
    model.load_state_dict(saved_ckt[model_name_in_ckt])

    return model
