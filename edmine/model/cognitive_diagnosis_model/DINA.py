import numpy as np
import torch
from torch import nn
import torch.autograd as autograd
import torch.nn.functional as F

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel


class DINA(nn.Module, DLCognitiveDiagnosisModel):
    model_name = "DINA"

    def __init__(self, params, objects):
        super(DINA, self).__init__()
        self.params = params
        self.objects = objects

        self.embed_layer = EmbedLayer(self.params["models_config"]["DINA"]["embed_config"])
        self.sign = StraightThroughEstimator()

    def forward(self, batch):
        model_config = self.params["models_config"]["DINA"]
        max_step = model_config["max_step"]
        max_slip = model_config["max_slip"]
        max_guess = model_config["max_guess"]
        use_ste = model_config["use_ste"]
        q_table = self.objects["dataset"]["q_table_tensor"]

        user_id = batch["user_id"]
        question_id = batch["question_id"]
        concept_one_hot = q_table[question_id]

        slip = torch.squeeze(torch.sigmoid(self.embed_layer.get_emb("slip", question_id)) * max_slip)
        guess = torch.squeeze(torch.sigmoid(self.embed_layer.get_emb("guess", question_id)) * max_guess)

        if use_ste:
            theta = self.sign(self.embed_layer.get_emb("theta", user_id))
            mask_theta = (concept_one_hot == 0) + (concept_one_hot == 1) * theta
            n = torch.prod((mask_theta + 1) / 2, dim=-1)

            return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)
        else:
            theta = self.embed_layer.get_emb("theta", user_id)
            if self.training:
                n = torch.sum(concept_one_hot * (torch.sigmoid(theta) - 0.5), dim=1)
                t, self.step = max((np.sin(2 * np.pi * self.step / max_step) + 1) / 2 * 100,
                                   1e-6), self.step + 1 if self.step < max_step else 0
                return torch.sum(
                    torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                    dim=1
                )
            else:
                n = torch.prod(concept_one_hot * (theta >= 0) + (1 - concept_one_hot), dim=1)
                return (1 - slip) ** n * guess ** (1 - n)

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


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x
