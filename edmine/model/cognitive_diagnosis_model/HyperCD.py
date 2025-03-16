import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.module.Clipper import NoneNegClipper
from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel
from edmine.model.loss import binary_cross_entropy


class HyperCD(nn.Module, DLCognitiveDiagnosisModel):
    model_name = "HyperCD"
    def __init__(self, params, objects):
        super(HyperCD, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["HyperCD"]
        num_user = model_config["num_user"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        dim_emb = model_config["dim_emb"]
        dim_feature = model_config["dim_feature"]
        self.student_emb = nn.Embedding(num_user, dim_emb, dtype=torch.float64)
        self.exercise_emb = nn.Embedding(num_question, dim_emb, dtype=torch.float64)
        self.knowledge_emb = nn.Embedding(num_concept, dim_emb, dtype=torch.float64)
        self.student_emb2feature = nn.Linear(dim_emb, dim_feature, dtype=torch.float64)
        self.exercise_emb2feature = nn.Linear(dim_emb, dim_feature, dtype=torch.float64)
        self.knowledge_emb2feature = nn.Linear(dim_emb, dim_feature, dtype=torch.float64)
        self.exercise_emb2discrimination = nn.Linear(dim_emb, 1, dtype=torch.float64)
        self.state2response = nn.Sequential(
            nn.Linear(num_concept, 512, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(128, 1, dtype=torch.float64),
            nn.Sigmoid()
        )

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def convolution(self, embedding, adj):
        num_layer = self.params["models_config"]["HyperCD"]["num_layer"]
        all_emb = embedding.weight.to(self.params["device"])
        final = [all_emb]
        for i in range(num_layer):
            # implement momentum hypergraph convolution
            all_emb = torch.sparse.mm(adj, all_emb) + 0.8 * all_emb
            final.append(all_emb)
        final_emb = torch.mean(torch.stack(final, dim=1), dim=1, dtype=torch.float64)
        return final_emb

    def forward(self, batch):
        leaky = self.params["models_config"]["HyperCD"]["leaky"]
        student_id = batch["user_id"]
        exercise_id = batch["question_id"]
        q_table = self.objects["dataset"]["q_table_tensor"]
        student_adj = self.objects["dataset"]["adj"]["user"]
        exercise_adj = self.objects["dataset"]["adj"]["question"]
        knowledge_adj = self.objects["dataset"]["adj"]["concept"]
        
        convolved_student_emb = self.convolution(self.student_emb, student_adj)
        convolved_exercise_emb = self.convolution(self.exercise_emb, exercise_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, knowledge_adj)

        batch_student = F.embedding(student_id, convolved_student_emb)
        batch_exercise = F.embedding(exercise_id, convolved_exercise_emb)

        student_feature = F.leaky_relu(self.student_emb2feature(batch_student), negative_slope=leaky)
        exercise_feature = F.leaky_relu(self.exercise_emb2feature(batch_exercise), negative_slope=leaky)
        knowledge_feature = F.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=leaky)
        discrimination = torch.sigmoid(self.exercise_emb2discrimination(batch_exercise))

        state = discrimination * (student_feature @ knowledge_feature.T
                                  - exercise_feature @ knowledge_feature.T) * q_table[exercise_id]

        predict_score = self.state2response(state).view(-1)
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
        for layer in self.state2response:
            if isinstance(layer, nn.Linear):
                layer.apply(clipper)


    def get_knowledge_state(self, user_id):
        leaky = self.params["models_config"]["HyperCD"]["leaky"]
        student_adj = self.objects["dataset"]["adj"]["user"]
        knowledge_adj = self.objects["dataset"]["adj"]["concept"]

        convolved_student_emb = self.convolution(self.student_emb, student_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, knowledge_adj)

        student_feature = F.leaky_relu(self.student_emb2feature(convolved_student_emb), negative_slope=leaky)
        knowledge_feature = F.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=leaky)

        return torch.sigmoid(student_feature @ knowledge_feature.T)[user_id].detach().cpu().numpy()

    def get_exercise_level(self):
        leaky = self.params["models_config"]["HyperCD"]["leaky"]
        exercise_adj = self.objects["dataset"]["adj"]["question"]
        knowledge_adj = self.objects["dataset"]["adj"]["concept"]
        
        convolved_exercise_emb = self.convolution(self.exercise_emb, exercise_adj)
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, knowledge_adj)

        exercise_feature = F.leaky_relu(self.exercise_emb2feature(convolved_exercise_emb), negative_slope=leaky)
        knowledge_feature = F.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=leaky)

        return torch.sigmoid(exercise_feature @ knowledge_feature.T).detach().cpu().numpy()

    def get_knowledge_feature(self):
        leaky = self.params["models_config"]["HyperCD"]["leaky"]
        convolved_knowledge_emb = self.convolution(self.knowledge_emb, self.knowledge_adj)
        knowledge_feature = F.leaky_relu(self.knowledge_emb2feature(convolved_knowledge_emb), negative_slope=leaky)
        return knowledge_feature.detach().cpu().numpy()