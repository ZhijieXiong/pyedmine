import numpy as np
import torch
import torch.nn as nn

from edmine.model.KnowledgeTracingModel import KnowledgeTracingModel
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import binary_cross_entropy


class TimeDualDecayEncoder(nn.Module):
    def __init__(self, dim_time, parameter_requires_grad=True):
        super(TimeDualDecayEncoder, self).__init__()
        self.time_dim = dim_time
        self.w_short = nn.Linear(1, dim_time)
        self.w_long = nn.Linear(1, dim_time)
        self.w_short.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time, dtype=np.float32))).reshape(dim_time, -1))
        self.w_short.bias = nn.Parameter(torch.zeros(dim_time))
        self.w_long.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time, dtype=np.float32))).reshape(dim_time, -1))
        self.w_long.bias = nn.Parameter(torch.zeros(dim_time))
        self.f = nn.ReLU()

        self.w_o = nn.Linear(dim_time, dim_time)
        self.w_o.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time*dim_time, dtype=np.float32))).reshape(dim_time, -1))
        self.w_o.bias = nn.Parameter(torch.zeros(dim_time))

        if not parameter_requires_grad:
            self.w_short.weight.requires_grad = False
            self.w_short.bias.requires_grad = False
            self.w_long.weight.requires_grad = False
            self.w_long.bias.requires_grad = False


    def forward(self, timestamps):
        timestamps = timestamps.unsqueeze(dim=2)
        timestamps_right = timestamps.clone()
        timestamps_right = torch.cat([timestamps_right[:,1:,:], timestamps_right[:,-1,:].unsqueeze(1)],dim=1)
        timestamps_diff = timestamps_right - timestamps

        timestamps_mask = (timestamps_diff > 3600*24).float()

        timestamps_short = self.f(self.w_short(timestamps_diff*timestamps_mask)) # torch.exp(-1*self.f(self.w_short(timestamps_diff*timestamps_mask)))
        timestamps_long = self.f(self.w_long(timestamps_diff*(1-timestamps_mask))) #torch.exp(-1*self.f(self.w_long(timestamps_diff*(1-timestamps_mask))))
        output = self.w_o(timestamps_short+timestamps_long) # -1#torch.exp(-1*self.f(self.w(timestamps)))

        return output
    

class DyGKT(nn.Module, KnowledgeTracingModel):
    def __init__(self, params, objects
                #  , node_raw_features: np.ndarray, edge_raw_features: np.ndarray
                 ):
        super(DyGKT, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["DyGKT"]
        dim_emb = model_config["dim_emb"]
        dim_time = model_config["dim_time"]
        
        self.performance_encoder = nn.Linear(1, 64)
        self.dual_time_encoder = TimeDualDecayEncoder(dim_time)
        self.multiset_indicator = nn.Linear(1, 64)
        self.gru_linear4user = nn.Linear(dim_emb, 64)
        self.gru4user = nn.GRU(dim_emb, 64)
        self.gru_linear4que = nn.Linear(dim_emb, 64)
        self.gru4que = nn.GRU(dim_emb, 64)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])
        
    def get_user_que_embedding(self, batch):
        X_se = self.performance_encoder(batch["user_history_correctness_seq"])
        X_qe = self.performance_encoder(batch["que_history_correctness_seq"])
        X_st = self.dual_time_encoder(batch["user_history_time_seq"])
        X_qt = self.dual_time_encoder(batch["que_history_time_seq"])
        
        
    def get_predict_score(self, batch, seq_start=2):
        pass
        
    def get_predict_loss(self, batch, seq_start=2):
        src_node_embeddings, dst_node_embeddings = self.get_user_que_embedding(batch)
        pass
        
    def get_knowledge_state(self, batch):
        pass


class DyKT_Seq(nn.Module):
    def __init__(self, edge_dim ,node_dim):
        super(DyKT_Seq,self).__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim,batch_first=True)# LSTM

    def update(self, x):
        outputs, _ = self.hid_node_updater(x)
        return torch.squeeze(outputs,dim=0)

    