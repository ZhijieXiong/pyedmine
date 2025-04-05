import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import bipartite_subgraph

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class SAGEConv4GIKT(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="mean", normalize=False, root_weight=True, project=False,
                 bias=True, **kwargs):
        # GIKT中用的公式是dropout(MLP(x_self + mean(x_neighbors)))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = nn.LSTM(in_channels[0], in_channels[0], batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        # gitk的代码里在消息传递——MLP(x_self+mean(x_neighbors))——时加了一个dropout
        self.dropout_gnn = nn.Dropout(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.aggr_module.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        # out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            # out就是邻居聚合, x_r是self节点
            if self.training:
                out = self.dropout_gnn(self.lin_r(out + x_r))
            else:
                out = self.lin_r(out + x_r)
            # out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t, x):
        adj_t = adj_t.set_value(None, layout=None)
        return torch.sparse.mm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


# pyg实现
class GIKT_PYG(nn.Module, DLSequentialKTModel):
    model_name = "GIKT"

    def __init__(self, params, objects):
        super(GIKT_PYG, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["GIKT"]
        dim_emb = model_config["dim_emb"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        dropout4gru = model_config["dropout4gru"]

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_correct = nn.Embedding(2, dim_emb)

        self.gru1 = nn.GRUCell(dim_emb * 2, dim_emb)
        self.gru2 = nn.GRUCell(dim_emb, dim_emb)
        self.gcn1 = SAGEConv4GIKT(dim_emb, dim_emb, flow="target_to_source")
        self.gcn2 = SAGEConv4GIKT(dim_emb, dim_emb, flow="target_to_source")
        self.gcn3 = SAGEConv4GIKT(dim_emb, dim_emb, flow="target_to_source")
        self.MLP_AGG_last = Linear(dim_emb, dim_emb)
        self.dropout_gru = nn.Dropout(dropout4gru)
        self.MLP_query = Linear(dim_emb, dim_emb)
        self.MLP_key = Linear(dim_emb, dim_emb)
        self.MLP_W = Linear(2 * dim_emb, 1)

    def forward(self, batch):
        model_config = self.params["models_config"]["GIKT"]
        dim_emb = model_config["dim_emb"]
        num_question = model_config["num_question"]
        num_concept = model_config["num_concept"]
        rank_k = model_config["rank_k"]

        edge_index_q2c = self.objects["gikt"]["edge_index"][0]
        edge_attr_q2c = self.objects["gikt"]["edge_attr"][0]
        edge_index_c2q = self.objects["gikt"]["edge_index"][1]
        edge_attr_c2q = self.objects["gikt"]["edge_attr"][1]

        question_seq = batch["question_seq"]
        correct_seq = batch["correct_seq"]
        mask_seq = batch["mask_seq"]

        batch_size, seq_len = question_seq.shape
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(dim_emb).repeat(batch_size, 1)).to(self.params["device"])
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(dim_emb).repeat(batch_size, 1)).to(self.params["device"])
        state_history = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        y_hat = torch.zeros(batch_size, seq_len).to(self.params["device"])
        all_emb_question = self.embed_question(
            torch.tensor([i for i in range(num_question)]).long().to(self.params["device"])
        )
        all_emb_concept = self.embed_concept(
            torch.tensor([i for i in range(num_concept)]).long().to(self.params["device"])
        )
        x = torch.concat((all_emb_concept, all_emb_question), dim=0)
        if self.training:
            selected_question_nodes = torch.tensor([]).long().to(self.params["device"])
        for t in range(seq_len-1):
            question_t = question_seq[:, t]
            response_t = correct_seq[:, t]
            mask_t = torch.ne(mask_seq[:, t], 0)
            emb_response_t = self.embed_correct(response_t)

            if self.training:
                selected_question_nodes = torch.concat((selected_question_nodes, question_t[mask_t]))
                # 如果每个step都做GCN，显存不够
                if t % 2 == 0:
                    selected_question_nodes = torch.unique(selected_question_nodes).clone().detach()
                    selected_neighbor_nodes = self.objects["data"]["Q_table_tensor"][selected_question_nodes]
                    # x的前self.num_concept行是知识点
                    selected_question_nodes = selected_question_nodes + num_concept
                    selected_neighbor_nodes = torch.unique(torch.nonzero(selected_neighbor_nodes == 1)[:, 1])
                    subset_q2c = (selected_question_nodes, torch.arange(num_concept).to(self.params["device"]))
                    subset_c2q = (selected_neighbor_nodes, torch.arange(num_question).to(self.params["device"]))
                    subgraph_edge_index_q2c, subgraph_edge_attr_q2c = \
                        bipartite_subgraph(subset_q2c, edge_index_q2c, edge_attr_q2c)
                    subgraph_edge_index_c2q, subgraph_edge_attr_c2q = \
                        bipartite_subgraph(subset_c2q, edge_index_c2q, edge_attr_c2q)
                    # GNN获得习题的embedding
                    x = self.gcn1(x, subgraph_edge_index_q2c)
                    x = self.gcn2(x, subgraph_edge_index_c2q)
                    x = self.gcn3(x, subgraph_edge_index_q2c)
                    x = torch.tanh(self.MLP_AGG_last(x))
                    selected_question_nodes = torch.tensor([]).long().to(self.params["device"])
            else:
                selected_question_nodes = torch.unique(question_t[mask_t])
                selected_neighbor_nodes = self.objects["data"]["Q_table_tensor"][selected_question_nodes]
                selected_question_nodes = selected_question_nodes + num_concept
                selected_neighbor_nodes = torch.unique(torch.nonzero(selected_neighbor_nodes == 1)[:, 1])
                subset_q2c = (selected_question_nodes, torch.arange(num_concept).to(self.params["device"]))
                subset_c2q = (selected_neighbor_nodes, torch.arange(num_question).to(self.params["device"]))
                subgraph_edge_index_q2c, subgraph_edge_attr_q2c = \
                    bipartite_subgraph(subset_q2c, edge_index_q2c, edge_attr_q2c)
                subgraph_edge_index_c2q, subgraph_edge_attr_c2q = \
                    bipartite_subgraph(subset_c2q, edge_index_c2q, edge_attr_c2q)
                # GNN获得习题的embedding
                x = self.gcn1(x, subgraph_edge_index_q2c)
                x = self.gcn2(x, subgraph_edge_index_c2q)
                x = self.gcn3(x, subgraph_edge_index_q2c)
                x = torch.tanh(self.MLP_AGG_last(x))
            emb_question_t = x[question_t[mask_t] + num_concept]
            emb_question_t_reconstruct = torch.zeros(batch_size, dim_emb).to(self.params["device"])
            emb_question_t_reconstruct[mask_t] = emb_question_t
            emb_question_t_reconstruct[~mask_t] = self.embed_question(question_t[~mask_t])

            # GRU更新知识状态
            gru1_input = torch.concat((emb_question_t_reconstruct, emb_response_t), dim=1)
            h1_pre = self.dropout_gru(self.gru1(gru1_input, h1_pre))
            gru2_output = self.dropout_gru(self.gru2(h1_pre, h2_pre))

            # 找t+1时刻习题对应的知识点
            question_next = question_seq[:, t + 1]
            correspond_concepts = self.objects["data"]["Q_table_tensor"][question_next]
            correspond_concepts_list = []
            max_concept = 1
            for i in range(batch_size):
                concepts_index = torch.nonzero(correspond_concepts[i] == 1).squeeze()
                if len(concepts_index.shape) == 0:
                    correspond_concepts_list.append(torch.unsqueeze(self.embed_concept(concepts_index), dim=0))
                else:
                    if concepts_index.shape[0] > max_concept:
                        max_concept = concepts_index.shape[0]
                    correspond_concepts_list.append(self.embed_concept(concepts_index))
            # 将习题和对应知识点embedding拼接起来
            emb_question_next = self.embed_question(question_next)
            question_concept = torch.zeros(batch_size, max_concept+1, dim_emb)
            for b, emb_concepts in enumerate(correspond_concepts_list):
                num_qc = 1 + emb_concepts.shape[0]
                emb_next = torch.unsqueeze(emb_question_next[b], dim=0)
                question_concept[b, 0:num_qc] = torch.concat((emb_next, emb_concepts), dim=0)
            question_concept = question_concept.to(self.params["device"])
            if t == 0:
                y_hat[:, 0] = self.predict(question_concept, torch.unsqueeze(gru2_output, dim=1))
                continue

            current_state = gru2_output.unsqueeze(dim=1)
            if t <= rank_k:
                current_history_state = torch.concat((current_state, state_history[:, 0:t]), dim=1)
            else:
                Q = self.embed_question(question_next).clone().detach().unsqueeze(dim=-1)
                K = self.embed_question(question_seq[:, 0:t]).clone().detach()
                product_score = torch.bmm(K, Q).squeeze(dim=-1)
                _, indices = torch.topk(product_score, k=rank_k, dim=1)
                select_history = torch.concat(tuple(state_history[i][indices[i]].unsqueeze(dim=0)
                                                    for i in range(batch_size)), dim=0)
                current_history_state = torch.concat((current_state, select_history), dim=1)

            y_hat[:, t + 1] = self.predict(question_concept, current_history_state)
            h2_pre = gru2_output
            state_history[:, t] = gru2_output
        return y_hat

    def get_predict_score(self, batch):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = self.forward(batch)
        predict_score = torch.masked_select(predict_score[:, 1:], mask_bool_seq[:, 1:])

        return predict_score

    def get_predict_loss(self, batch, loss_record=None):
        mask_bool_seq = torch.ne(batch["mask_seq"], 0)

        predict_score = self.get_predict_score(batch)
        ground_truth = torch.masked_select(batch["correct_seq"][:, 1:], mask_bool_seq[:, 1:])
        predict_loss = nn.functional.binary_cross_entropy(predict_score.double(), ground_truth.double())

        if loss_record is not None:
            num_sample = torch.sum(batch["mask_seq"][:, 1:]).item()
            loss_record.add_loss("predict loss", predict_loss.detach().cpu().item() * num_sample, num_sample)

        return predict_loss

    def predict(self, question_concept, current_history_state):
        # question_concept: (batch_size, num_qc, dim_emb), current_history_state: (batch_size, num_state, dim_emb)
        output_g = torch.bmm(question_concept, torch.transpose(current_history_state, 1, 2))

        num_qc, num_state = question_concept.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(current_history_state, dim=1)  # [batch_size, 1, num_state, dim_emb]
        states = states.repeat(1, num_qc, 1, 1)  # [batch_size, num_qc, num_state, dim_emb]
        question_concepts = torch.unsqueeze(question_concept, dim=2)  # [batch_size, num_qc, 1, dim_emb]
        question_concepts = question_concepts.repeat(1, 1, num_state, 1)  # [batch_size, num_qc, num_state, dim_emb]

        K = torch.tanh(self.MLP_query(states))  # [batch_size, num_qc, num_state, dim_emb]
        Q = torch.tanh(self.MLP_key(question_concepts))  # [batch_size, num_qc, num_state, dim_emb]
        tmp = self.MLP_W(torch.concat((Q, K), dim=-1))  # [batch_size, num_qc, num_state, 1]
        tmp = torch.squeeze(tmp, dim=-1)  # [batch_size, num_qc, num_state]
        alpha = torch.softmax(tmp, dim=2)  # [batch_size, num_qc, num_state]
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)  # [batch_size, 1]
        result = torch.sigmoid(torch.squeeze(p, dim=-1))
        return result
