import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class NCD(nn.Module, DLCognitiveDiagnosisModel):
    model_name = "NCD"

    def __init__(self, params, objects):
        super(NCD, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["NCD"]
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def forward(self, batch):
        user_id = batch["user_id"]
        question_id = batch["question_id"]
        Q_table = self.objects["dataset"]["q_table_tensor"]

        user_emb = torch.sigmoid(self.embed_layer.get_emb("user", user_id))
        question_diff = torch.sigmoid(self.embed_layer.get_emb("question_diff", question_id))
        question_disc = torch.sigmoid(self.embed_layer.get_emb("question_disc", question_id)) * 10
        predict_layer_input = question_disc * (user_emb - question_diff) * Q_table[question_id]
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_loss(self, batch):
        predict_score = self.forward(batch)
        ground_truth = batch["correctness"]
        if self.params["device"] == "mps":
            loss = torch.nn.functional.binary_cross_entropy(predict_score.float(), ground_truth.float())
        else:
            loss = torch.nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())
        num_sample = batch["correctness"].shape[0]
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
            },
            "predict_score": predict_score
        }

    def get_predict_score(self, batch):
        predict_score = self.forward(batch)
        return {
            "predict_score": predict_score,
        }

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.predict_layer1.apply(clipper)
        self.predict_layer2.apply(clipper)
        self.predict_layer3.apply(clipper)

    def get_knowledge_state(self, user_id):
        return torch.sigmoid(self.embed_layer.get_emb("user", user_id))
    