import torch
import torch.nn as nn
import torch.nn.functional as F
    
from edmine.model.module.Graph import RCDGraphLayer
from edmine.model.module.Clipper import NoneNegClipper
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel
from edmine.model.loss import binary_cross_entropy
            
            
class Fusion(nn.Module):
    def __init__(self, params, objects):
        super(Fusion, self).__init__()
        self.params = params
        self.objects = objects
        
        num_concept = params["models_config"]["RCD"]["embed_config"]["concept"]["num_item"]
        directed_g = objects["dataset"]["local_map"]['directed_g']
        undirected_g = objects["dataset"]["local_map"]['undirected_g']
        k_from_e = objects["dataset"]["local_map"]['k_from_e']
        e_from_k = objects["dataset"]["local_map"]['e_from_k']
        u_from_e = objects["dataset"]["local_map"]['u_from_e']
        e_from_u = objects["dataset"]["local_map"]['e_from_u']

        self.directed_gat = RCDGraphLayer(directed_g, num_concept, num_concept)
        self.undirected_gat = RCDGraphLayer(undirected_g, num_concept, num_concept)

        self.k_from_e = RCDGraphLayer(k_from_e, num_concept, num_concept)  # src: e
        self.e_from_k = RCDGraphLayer(e_from_k, num_concept, num_concept)  # src: k

        self.u_from_e = RCDGraphLayer(u_from_e, num_concept, num_concept)  # src: e
        self.e_from_u = RCDGraphLayer(e_from_u, num_concept, num_concept)  # src: u

        self.k_attn_fc1 = nn.Linear(2 * num_concept, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * num_concept, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * num_concept, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * num_concept, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * num_concept, 1, bias=True)

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        num_question = self.params["models_config"]["RCD"]["embed_config"]["question"]["num_item"]
        
        k_directed = self.directed_gat(kn_emb)
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        k_from_e_graph = self.k_from_e(e_k_graph)
        e_from_k_graph = self.e_from_k(e_k_graph)

        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_graph = self.u_from_e(e_u_graph)
        e_from_u_graph = self.e_from_u(e_u_graph)

        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        D = k_from_e_graph[num_question:]
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        concat_c_3 = torch.cat([A, D], dim=1)
        score1 = self.k_attn_fc1(concat_c_1)
        score2 = self.k_attn_fc2(concat_c_2)
        score3 = self.k_attn_fc3(concat_c_3)
        score = F.softmax(torch.cat([torch.cat([score1, score2], dim=1), score3], dim=1),
                          dim=1)  # dim = 1, 按行SoftMax, 行和为1
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C + score[:, 2].unsqueeze(1) * D

        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0: num_question]
        C = e_from_u_graph[0: num_question]
        concat_e_1 = torch.cat([A, B], dim=1)
        concat_e_2 = torch.cat([A, C], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        score2 = self.e_attn_fc2(concat_e_2)
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)  # dim = 1, 按行SoftMax, 行和为1
        exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

        # updated students
        all_stu_emb = all_stu_emb + u_from_e_graph[num_question:]

        return kn_emb, exer_emb, all_stu_emb
    
    
class RCD(nn.Module, DLCognitiveDiagnosisModel):
    model_name = "RCD"
    def __init__(self, params, objects):
        super(RCD, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = params["models_config"]["RCD"]
        num_concept = model_config["embed_config"]["concept"]["num_item"]
        num_question = model_config["embed_config"]["question"]["num_item"]
        num_user = model_config["embed_config"]["user"]["num_item"]
        
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.k_index = torch.LongTensor(list(range(num_concept))).to(params["device"])
        self.stu_index = torch.LongTensor(list(range(num_user))).to(params["device"])
        self.exer_index = torch.LongTensor(list(range(num_question))).to(params["device"])
        self.FusionLayer1 = Fusion(params, objects)
        self.FusionLayer2 = Fusion(params, objects)
        self.prednet_full1 = nn.Linear(2 * num_concept, num_concept, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * num_concept, num_concept, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * num_concept, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, batch):
        user_id = batch["user_id"]
        question_id = batch["question_id"]
        q_table = self.objects["dataset"]["q_table_tensor"]
        
        all_user_emb = self.embed_layer.get_emb("user", self.stu_index)
        all_question_emb = self.embed_layer.get_emb("question", self.exer_index)
        all_concept_emb = self.embed_layer.get_emb("concept", self.k_index)

        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(all_concept_emb, all_question_emb, all_user_emb)
        # Fusion layer 2
        kn_emb2, exer_emb2, all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

        # get batch student data
        batch_stu_emb = all_stu_emb2[user_id] # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = exer_emb2[question_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb2.shape[0], kn_emb2.shape[1])

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * q_table[question_id].unsqueeze(2), dim = 1)
        count_of_concept = torch.sum(q_table[question_id], dim = 1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output.squeeze(-1)
    
    def get_predict_score(self, batch):
        predict_score = self.forward(batch)
        return {
            "predict_score": predict_score,
        }
    
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

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
        
    def get_knowledge_state(self, user_id):
        return torch.sigmoid(self.embed_layer.get_emb("user", user_id)).detach().cpu().numpy()
