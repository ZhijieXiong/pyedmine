import torch
from torch import nn
import torch.nn.functional as F

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel


class IRT(nn.Module, DLCognitiveDiagnosisModel):
    model_name = "IRT"

    def __init__(self, params, objects):
        super(IRT, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = EmbedLayer(self.params["models_config"]["IRT"]["embed_config"])

    def forward(self, batch):
        user_id = batch["user_id"]
        question_id = batch["question_id"]

        model_config = self.params["models_config"]["IRT"]
        value_range = model_config["value_range"]
        a_range = model_config["a_range"]
        D = model_config["D"]

        theta = torch.squeeze(self.embed_layer.get_emb("theta", user_id), dim=-1)
        a = torch.squeeze(self.embed_layer.get_emb("a", question_id), dim=-1)
        b = torch.squeeze(self.embed_layer.get_emb("b", question_id), dim=-1)
        c = torch.sigmoid(torch.squeeze(self.embed_layer.get_emb("c", question_id), dim=-1))
        if value_range > 0:
            theta = value_range * (torch.sigmoid(theta) - 0.5)
            b = value_range * (torch.sigmoid(b) - 0.5)
        if a_range > 0:
            a = a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)

        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')

        return c + (1 - c) / (1 + torch.exp(-D * a * (theta - b)))

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
