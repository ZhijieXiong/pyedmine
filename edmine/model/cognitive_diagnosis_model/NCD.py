import torch
import torch.nn as nn

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel
from edmine.model.module.Clipper import NoneNegClipper
from edmine.model.loss import binary_cross_entropy


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
        q_table = self.objects["dataset"]["q_table_tensor"]

        user_emb = torch.sigmoid(self.embed_layer.get_emb("user", user_id))
        question_diff = torch.sigmoid(self.embed_layer.get_emb("question_diff", question_id))
        question_disc = torch.sigmoid(self.embed_layer.get_emb("question_disc", question_id)) * 10
        predict_layer_input = question_disc * (user_emb - question_diff) * q_table[question_id]
        predict_score = self.predict_layer(predict_layer_input).squeeze(dim=-1)

        return predict_score

    def get_predict_loss(self, batch):
        predict_score = self.forward(batch)
        ground_truth = batch["correctness"]
        loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
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
        for layer in self.predict_layer.predict_layer:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)

    def get_knowledge_state(self, user_id):
        return torch.sigmoid(self.embed_layer.get_emb("user", user_id))
    