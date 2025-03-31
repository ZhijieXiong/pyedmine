import numpy as np
import torch
import torch.nn as nn

from edmine.model.KnowledgeTracingModel import KnowledgeTracingModel
from edmine.model.module.PredictorLayer import PredictorLayer


class TimeDecayEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeDecayEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))
        self.f = nn.ReLU()

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        timestamps = timestamps.unsqueeze(dim=2)
        # shape (batch_size, seq_len, time_dim)
        output = torch.exp(-1*self.f(self.w(timestamps)))
        return output


class TimeDualDecayEncoder(nn.Module):
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeDualDecayEncoder, self).__init__()
        self.time_dim = time_dim
        self.o_encode = TimeDecayEncoder(time_dim)
        self.w_short = nn.Linear(1, time_dim)
        self.w_long = nn.Linear(1, time_dim)
        self.w_short.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w_short.bias = nn.Parameter(torch.zeros(time_dim))
        self.w_long.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w_long.bias = nn.Parameter(torch.zeros(time_dim))
        self.f = nn.ReLU()

        self.w_o = nn.Linear(time_dim, time_dim)
        self.w_o.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim*time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w_o.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w_short.weight.requires_grad = False
            self.w_short.bias.requires_grad = False
            self.w_long.weight.requires_grad = False
            self.w_long.bias.requires_grad = False


    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        timestamps = timestamps.unsqueeze(dim=2)
        timestamps_right = timestamps.clone()
        timestamps_right = torch.cat([timestamps_right[:,1:,:], timestamps_right[:,-1,:].unsqueeze(1)],dim=1)
        timestamps_diff = timestamps_right - timestamps

        timestamps_mask = (timestamps_diff > 3600*24).float()

        timestamps_short = self.f(self.w_short(timestamps_diff*timestamps_mask)) # torch.exp(-1*self.f(self.w_short(timestamps_diff*timestamps_mask)))
        timestamps_long = self.f(self.w_long(timestamps_diff*(1-timestamps_mask))) #torch.exp(-1*self.f(self.w_long(timestamps_diff*(1-timestamps_mask))))
        # shape (batch_size, seq_len, time_dim)
        output = self.w_o(timestamps_short+timestamps_long) # -1#torch.exp(-1*self.f(self.w(timestamps)))

        return output # + o_output
    

class DyGKT(nn.Module, KnowledgeTracingModel):
    def __init__(self, params, objects, node_raw_features: np.ndarray, edge_raw_features: np.ndarray):
        super(DyGKT, self).__init__()
        self.params = params
        self.objects = objects
        
        model_config = self.params["models_config"]["DyGKT"]
        num_concept = model_config["num_concept"]
        dim_time = model_config["dim_time"]
        
        # node_raw_features:每一个interaction中q所属的知识点（只能在单知识点数据集上跑）
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        # edge_raw_features:每一个interaction的标签
        # self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))
        self.projection_layer = nn.ModuleDict({
            'feature_Linear':nn.Linear(in_features=self.node_raw_features.shape[-1], out_features=64, bias=True),
            'edge': nn.Linear(in_features=1, out_features=64, bias=True),
            'time': nn.Linear(in_features=dim_time, out_features=64, bias=True),
            'struct': nn.Linear(in_features=1, out_features=64, bias=True),
        })
        self.output_layer = nn.Linear(in_features=64, out_features=64, bias=True)
        self.src_node_updater = DyKT_Seq(edge_dim=64, node_dim=64)
        self.dst_node_updater = DyKT_Seq(edge_dim=64, node_dim=64)
        self.time_encoder = TimeDualDecayEncoder(time_dim=dim_time)
        self.predict_layer = PredictorLayer(model_config["predictor_config"])

    def compute_src_dst_node_temporal_embeddings(self, batch):
        num_neighbor = self.params["models_config"]["DyGKT"]["num_neighbor"]
        
        src_node_ids = batch["src_node"]
        node_interact_times = batch["time"]
        dst_node_ids = batch["dst_node"]
        correctness = batch["correctness"]
        batch_size = src_node_ids.shape[0]
        
        neighbor_src_node_ids = batch["neighbor_src_node"]
        neighbor_src_edge_ids = batch["neighbor_src_edge"]
        neighbor_src_times = batch["neighbor_src_time"]
        
        neighbor_dst_node_ids = batch["neighbor_dst_node"]
        neighbor_dst_edge_ids = batch["neighbor_dst_edge"]
        neighbor_dst_times = batch["neighbor_dst_time"]
        
        neighbor_src_node_ids = torch.cat((neighbor_src_node_ids, src_node_ids.unsqueeze(1)), dim=-1)
        neighbor_src_edge_ids = torch.cat((neighbor_src_edge_ids, torch.zeros((batch_size, 1)).to(self.params["device"])), dim=-1)
        neighbor_src_times = torch.cat((neighbor_src_times, node_interact_times.unsqueeze(1)), dim=-1)
        
        neighbor_dst_node_ids = torch.cat((neighbor_dst_node_ids, dst_node_ids.unsqueeze(1)), dim=-1)
        neighbor_dst_edge_ids = torch.cat((neighbor_dst_edge_ids, torch.zeros((batch_size, 1)).to(self.params["device"])), dim=-1)
        neighbor_dst_times = torch.cat((neighbor_dst_times, node_interact_times.unsqueeze(1)), dim=-1)

        src_nodes_neighbor_co_occurrence_features = (neighbor_src_node_ids[:,:-1] == dst_node_ids.unsqueeze(1).repeat(1, num_neighbor)).unsqueeze(-1).float()
        dst_nodes_neighbor_co_occurrence_features = (neighbor_dst_node_ids[:,:-1] == src_node_ids.unsqueeze(1).repeat(1, num_neighbor)).unsqueeze(-1).float()

        src_node_skill = self.node_raw_features[neighbor_src_node_ids][:, :-1, 0]
        dst_node_skill = self.node_raw_features[neighbor_dst_node_ids][:, -1, 0].unsqueeze(1).repeat(1, num_neighbor)

        src_nodes_neighbor_skill_features = (src_node_skill == dst_node_skill).unsqueeze(-1).float()
        a = 1
        
        src_nodes_neighbor_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_co_occurrence_features)
        dst_nodes_neighbor_struct_features = self.projection_layer['struct'](a * dst_nodes_neighbor_co_occurrence_features)
        src_nodes_neighbor_skill_struct_features = self.projection_layer['struct'](a * src_nodes_neighbor_skill_features)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=neighbor_src_edge_ids,
            nodes_neighbor_ids=neighbor_src_node_ids, nodes_neighbor_times=neighbor_src_times)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = self.get_features(
            node_interact_times=node_interact_times, nodes_edge_ids=neighbor_dst_edge_ids,
            nodes_neighbor_ids=neighbor_dst_node_ids, nodes_neighbor_times=neighbor_dst_times)
        
        src_nodes_features = src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features 
        dst_nodes_features = dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features 

        src_node_embeddings = self.src_node_updater.update(
            src_nodes_features[:, :-1, :] + src_nodes_neighbor_skill_struct_features+ src_nodes_neighbor_struct_features) + (src_nodes_edge_raw_features + src_nodes_neighbor_time_features)[:,-1, :]
        dst_node_embeddings = self.dst_node_updater.update((dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)[:, :-1, :]+ dst_nodes_neighbor_struct_features) + dst_nodes_features[:,-1,:] 
        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)
        
        return src_node_embeddings, dst_node_embeddings


    def get_features(self, node, edge, time):
        nodes_neighbor_node_raw_features = self.projection_layer['feature_Linear'](self.node_raw_features[node]) 
        nodes_neighbor_time_features = self.time_encoder(time)
        nodes_neighbor_time_features = self.projection_layer['time'](nodes_neighbor_time_features)
        nodes_edge_raw_features = self.projection_layer['edge'](edge)[:,:,0].unsqueeze(-1)
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features
    
class DyKT_Seq(nn.Module):
    def __init__(self, edge_dim : int,node_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(DyKT_Seq,self).__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim,batch_first=True)# LSTM

    def update(self, x):
        outputs, _ = self.hid_node_updater(x)
        return torch.squeeze(outputs,dim=0)


class MergeLayer(nn.Module):
    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        """
        Merge Layer to merge two inputs via: input_dim1 + input_dim2 -> hidden_dim -> output_dim.
        :param input_dim1: int, dimension of first input
        :param input_dim2: int, dimension of the second input
        :param hidden_dim: int, hidden dimension
        :param output_dim: int, dimension of the output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        merge and project the inputs
        :param input_1: Tensor, shape (*, input_dim1)
        :param input_2: Tensor, shape (*, input_dim2)
        :return:
        """
        # Tensor, shape (*, input_dim1 + input_dim2)
        # print(".shape.shape.shape.shape",input_1.shape, input_2.shape)
        x = torch.cat([input_1, input_2], dim=1)
        # Tensor, shape (*, output_dim)
        h = self.fc2(self.act(self.fc1(x)))
        return h