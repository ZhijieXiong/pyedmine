import torch
import networkx as nx
import numpy as np
import torch.nn as nn

from edmine.model.cognitive_diagnosis_model.DLCognitiveDiagnosisModel import DLCognitiveDiagnosisModel
from edmine.model.loss import binary_cross_entropy


def irt2pl(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return 1 / (1 + torch.exp(-1.7*item_offset*(user_emb - item_emb) ))

def mirt2pl(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return 1 / (1 + torch.exp(- torch.sum(torch.mul(user_emb, item_emb), axis=1).reshape(-1,1) + item_offset))

def sigmoid_dot(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return torch.sigmoid(torch.sum(torch.mul(user_emb, item_emb), axis = -1)).reshape(-1,1)

def dot(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return torch.sum(torch.mul(user_emb, item_emb), axis = -1).reshape(-1,1)

itf_dict = {
    'irt': irt2pl,
    'mirt': mirt2pl,
    'mf': dot, 
    'sigmoid-mf': sigmoid_dot
}


class HierCDF(nn.Module, DLCognitiveDiagnosisModel):
    model_name = "HierCDF"
    def __init__(self, params, objects):
        super(HierCDF, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["HierCDF"]
        num_user = model_config["num_user"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        dim_hidden = model_config["dim_hidden"]
        itf_type = model_config["itf_type"]
        know_graph = self.objects["HierCDF"]["know_graph"]

        self.know_edge = nx.DiGraph()
        for k in range(num_concept):
            self.know_edge.add_node(k)
        for edge in know_graph.values.tolist():
            self.know_edge.add_edge(edge[0],edge[1])

        self.topo_order = list(nx.topological_sort(self.know_edge))

        # the conditional mastery degree when parent is mastered
        condi_p = torch.Tensor(num_user, know_graph.shape[0])
        self.condi_p = nn.Parameter(condi_p)

        # the conditional mastery degree when parent is non-mastered
        condi_n = torch.Tensor(num_user, know_graph.shape[0])
        self.condi_n = nn.Parameter(condi_n)

        # the priori mastery degree of parent
        priori = torch.Tensor(num_user, num_concept)
        self.priori = nn.Parameter(priori)

        # item representation
        self.item_diff = nn.Embedding(num_question, num_concept)
        self.item_disc = nn.Embedding(num_question, 1)

        # embedding transformation
        self.user_contract = nn.Linear(num_concept, dim_hidden)
        self.item_contract = nn.Linear(num_concept, dim_hidden)

        # Neural Interaction Module (used only in ncd)
        self.cross_layer1=nn.Linear(dim_hidden,max(int(dim_hidden/2),1))
        self.cross_layer2=nn.Linear(max(int(dim_hidden/2),1),1)

        # layer for featrue cross module
        self.set_itf(itf_type)

        # param initialization
        nn.init.xavier_normal_(self.priori)
        nn.init.xavier_normal_(self.condi_p)
        nn.init.xavier_normal_(self.condi_n)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def ncd(self, user_emb, item_emb, item_offset):
        input_vec = (user_emb-item_emb)*item_offset
        x_vec=torch.sigmoid(self.cross_layer1(input_vec))
        x_vec=torch.sigmoid(self.cross_layer2(x_vec))
        return x_vec
    
    def set_itf(self, itf_type):
        self.itf_type = itf_type
        self.itf = itf_dict.get(itf_type, self.ncd)

    def get_posterior(self, user_id):
        num_concept = self.params["models_config"]["HierCDF"]["num_concept"]
        know_graph = self.objects["HierCDF"]["know_graph"]
        
        n_batch = user_id.shape[0]
        posterior = torch.rand(n_batch, num_concept).to(self.params["device"])
        batch_priori = torch.sigmoid(self.priori[user_id,:])
        batch_condi_p = torch.sigmoid(self.condi_p[user_id,:])
        batch_condi_n = torch.sigmoid(self.condi_n[user_id,:])
        
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)

            # for each knowledge k, do:
            if len_p == 0:
                priori = batch_priori[:,k]
                posterior[:,k] = priori.reshape(-1)
                continue

            # format of masks
            fmt = '{0:0%db}'%(len_p)
            # number of parent master condition
            n_condi = 2 ** len_p

            # sigmoid to limit priori to (0,1)
            #priori = batch_priori[:,predecessors]
            priori = posterior[:,predecessors]

            # self.logger.write('priori:{}'.format(priori.requires_grad),'console')

            pred_idx = know_graph[know_graph['to'] == k].sort_values(by='from').index
            condi_p = torch.pow(batch_condi_p[:,pred_idx],1/len_p)
            condi_n = torch.pow(batch_condi_n[:,pred_idx],1/len_p)
            
            margin_p = condi_p * priori
            margin_n = condi_n * (1.0-priori)

            posterior_k = torch.zeros((1,n_batch)).to(self.params["device"])

            for idx in range(n_condi):
                # for each parent mastery condition, do:
                mask = fmt.format(idx)
                mask = torch.Tensor(np.array(list(mask)).astype(int)).to(self.params["device"])

                margin = mask * margin_p + (1-mask) * margin_n
                margin = torch.prod(margin, dim = 1).unsqueeze(dim = 0)

                posterior_k = torch.cat([posterior_k, margin], dim = 0)
            posterior_k = (torch.sum(posterior_k, dim = 0)).squeeze()
            
            posterior[:,k] = posterior_k.reshape(-1)

        return posterior
    
    def get_condi_p(self, user_id):
        num_concept = self.params["models_config"]["HierCDF"]["num_concept"]
        know_graph = self.objects["HierCDF"]["know_graph"]
        
        n_batch = user_id.shape[0]
        result_tensor = torch.rand(n_batch, num_concept).to(self.params["device"])
        batch_priori = torch.sigmoid(self.priori[user_id,:])
        batch_condi_p = torch.sigmoid(self.condi_p[user_id,:])
        
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:,k]
                result_tensor[:,k] = priori.reshape(-1)
                continue
            pred_idx = know_graph[know_graph['to'] == k].sort_values(by='from').index
            condi_p = torch.pow(batch_condi_p[:,pred_idx],1/len_p)
            result_tensor[:,k] = torch.prod(condi_p, dim=1).reshape(-1)
        
        return result_tensor

    def get_condi_n(self, user_id):
        num_concept = self.params["models_config"]["HierCDF"]["num_concept"]
        know_graph = self.objects["HierCDF"]["know_graph"]
        
        n_batch = user_id.shape[0]
        result_tensor = torch.rand(n_batch, num_concept).to(self.params["device"])
        batch_priori = torch.sigmoid(self.priori[user_id,:])
        batch_condi_n = torch.sigmoid(self.condi_n[user_id,:])
        
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.know_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:,k]
                result_tensor[:,k] = priori.reshape(-1)
                continue
            pred_idx = know_graph[know_graph['to'] == k].sort_values(by='from').index
            condi_n = torch.pow(batch_condi_n[:,pred_idx],1/len_p)
            result_tensor[:,k] = torch.prod(condi_n, dim=1).reshape(-1)
        
        return result_tensor

    def forward(self, batch):
        q_table = self.objects["dataset"]["q_table_tensor"]
        user_id = batch["user_id"]
        question_id = batch["question_id"]
        
        user_mastery = self.get_posterior(user_id)
        item_diff = torch.sigmoid(self.item_diff(question_id))
        item_disc = torch.sigmoid(self.item_disc(question_id))

        user_factor = torch.tanh(self.user_contract(user_mastery * q_table[question_id]))
        item_factor = torch.sigmoid(self.item_contract(item_diff * q_table[question_id]))
        
        output = self.itf(user_factor, item_factor, item_disc)

        return output 
    
    def get_predict_loss(self, batch):
        user_id = batch["user_id"]
        predict_score = self.forward(batch)
        ground_truth = batch["correctness"]
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        penalty_loss = torch.sum(torch.relu(self.condi_n[user_id,:]-self.condi_p[user_id,:]))
        loss = predict_loss + self.params["loss_config"]["penalty loss"] * penalty_loss
        
        num_sample = batch["correctness"].shape[0]
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "penalty loss": {
                    "value": penalty_loss.detach().cpu().item() * num_sample,
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
            
    def pos_clipper(self, module_list):
        for module in module_list:
            module.weight.data = module.weight.clamp_min(0)
        return
    
    def apply_clipper(self):
        self.pos_clipper([self.user_contract,self.item_contract])
        self.pos_clipper([self.cross_layer1,self.cross_layer2])
