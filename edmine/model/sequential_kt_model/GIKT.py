import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class GIKT(nn.Module, DLSequentialKTModel):
    model_name = "GIKT"

    def __init__(self, params, objects):
        super(GIKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["GIKT"]
        dim_emb = model_config["dim_emb"]
        num_question, num_concept = self.objects["dataset"]["q_table"].shape[0], \
            self.objects["dataset"]["q_table"].shape[1]
        agg_hops = model_config["agg_hops"]
        dropout4gru = model_config["dropout4gru"]
        dropout4gnn = model_config["dropout4gnn"]

        self.question_neighbors = objects["GIKT"]["question_neighbors"]
        self.concept_neighbors = objects["GIKT"]["concept_neighbors"]

        self.embed_question = nn.Embedding(num_question, dim_emb)
        self.embed_concept = nn.Embedding(num_concept, dim_emb)
        self.embed_correctness = nn.Embedding(2, dim_emb)

        self.gru1 = nn.GRUCell(dim_emb * 2, dim_emb)
        self.gru2 = nn.GRUCell(dim_emb, dim_emb)
        self.mlp4agg = nn.ModuleList(Linear(dim_emb, dim_emb) for _ in range(agg_hops))
        self.MLP_AGG_last = Linear(dim_emb, dim_emb)
        self.dropout_gru = nn.Dropout(dropout4gru)
        self.dropout_gnn = nn.Dropout(dropout4gnn)
        self.MLP_query = Linear(dim_emb, dim_emb)
        self.MLP_key = Linear(dim_emb, dim_emb)
        # 公式10中的W
        self.MLP_W = Linear(2 * dim_emb, 1)

    def forward(self, batch):
        model_config = self.params["models_config"]["GIKT"]
        dim_emb = model_config["dim_emb"]
        agg_hops = model_config["agg_hops"]
        rank_k = model_config["rank_k"]

        question_seq = batch["question_seq"]
        correctness_seq = batch["correctness_seq"]
        mask_seq = batch["mask_seq"]

        batch_size, seq_len = question_seq.shape
        q_neighbor_size, c_neighbor_size = self.question_neighbors.shape[1], self.concept_neighbors.shape[1]
        h1_pre = torch.nn.init.xavier_uniform_(torch.zeros(dim_emb).repeat(batch_size, 1)).to(self.params["device"])
        h2_pre = torch.nn.init.xavier_uniform_(torch.zeros(dim_emb).repeat(batch_size, 1)).to(self.params["device"])
        state_history = torch.zeros(batch_size, seq_len, dim_emb).to(self.params["device"])
        y_hat = torch.zeros(batch_size, seq_len).to(self.params["device"])

        for t in range(seq_len-1):
            question_t = question_seq[:, t]
            response_t = correctness_seq[:, t]
            mask_t = torch.ne(mask_seq[:, t], 0)
            emb_response_t = self.embed_correctness(response_t)

            # GNN获得习题的embedding
            nodes_neighbor = [question_t[mask_t]]
            batch_size__ = len(nodes_neighbor[0])
            for i in range(agg_hops):
                nodes_current = nodes_neighbor[-1]
                nodes_current = nodes_current.reshape(-1)
                neighbor_shape = [batch_size__] + \
                                 [(q_neighbor_size if j % 2 == 0 else c_neighbor_size) for j in range(i + 1)]
                # 找知识点节点
                if i % 2 == 0:
                    nodes_neighbor.append(self.question_neighbors[nodes_current].reshape(neighbor_shape))
                    continue
                # 找习题节点
                nodes_neighbor.append(self.concept_neighbors[nodes_current].reshape(neighbor_shape))
            emb_nodes_neighbor = []
            for i, nodes in enumerate(nodes_neighbor):
                if i % 2 == 0:
                    emb_nodes_neighbor.append(self.embed_question(nodes))
                    continue
                emb_nodes_neighbor.append(self.embed_concept(nodes))
            emb_question_t = self.aggregate(emb_nodes_neighbor)
            emb_question_t_reconstruct = torch.zeros(batch_size, dim_emb).to(self.params["device"])
            emb_question_t_reconstruct[mask_t] = emb_question_t
            emb_question_t_reconstruct[~mask_t] = self.embed_question(question_t[~mask_t])

            # GRU更新知识状态
            gru1_input = torch.concat((emb_question_t_reconstruct, emb_response_t), dim=1)
            h1_pre = self.dropout_gru(self.gru1(gru1_input, h1_pre))
            gru2_output = self.dropout_gru(self.gru2(h1_pre, h2_pre))

            # 找t+1时刻习题对应的知识点
            question_next = question_seq[:, t + 1]
            correspond_concepts = self.objects["dataset"]["q_table_tensor"][question_next]
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
            question_concept = torch.zeros(batch_size, max_concept + 1, dim_emb)
            for b, emb_concepts in enumerate(correspond_concepts_list):
                num_qc = 1 + emb_concepts.shape[0]
                emb_next = torch.unsqueeze(emb_question_next[b], dim=0)
                question_concept[b, 0:num_qc] = torch.concat((emb_next, emb_concepts), dim=0)
            question_concept = question_concept.to(self.params["device"])
            if t == 0:
                y_hat[:, 0] = self.predict(question_concept, torch.unsqueeze(gru2_output, dim=1))
                continue
            # recap选取历史状态
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

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])

        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def aggregate(self, emb_list):
        # 输入是节点（习题节点）的embedding，计算步骤是：将节点和邻居的embedding相加，再通过一个MLP输出（embedding维度不变），激活函数用的tanh
        # 假设聚合3跳，那么输入是[0,1,2,3]，分别表示输入节点，1跳节点，2跳节点，3跳节点，总共聚合3次
        # 第1次聚合（每次聚合使用相同的MLP），(0,1)聚合得到新的embedding，放到输入位置0上；然后(1,2)聚合得到新的embedding，放到输入位置1上；然后(2,3)聚合得到新的embedding，放到输入位置2上
        # 第2次聚合，(0',1')，聚合得到新的embedding，放到输入位置0上；然后(1',2')聚合得到新的embedding，放到输入位置1上
        # 第3次聚合，(0'',1'')，聚合得到新的embedding，放到输入位置0上
        # 最后0'''通过一个MLP得到最终的embedding
        # aggregate from outside to inside
        agg_hops = self.params["models_config"]["GIKT"]["agg_hops"]
        for i in range(agg_hops):
            for j in range(agg_hops - i):
                emb_list[j] = self.sum_aggregate(emb_list[j], emb_list[j+1], j)
        return torch.tanh(self.MLP_AGG_last(emb_list[0]))

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        emb_sum = emb_sum_neighbor + emb_self
        return torch.tanh(self.dropout_gnn(self.mlp4agg[hop](emb_sum)))

    def recap_hard(self, current_q, history_q):
        batch_size = current_q.shape[0]
        q_neighbor_size, c_neighbor_size = self.question_neighbors.shape[1], self.concept_neighbors.shape[1]
        nodes_current = current_q.reshape(-1)
        neighbors_concept = self.question_neighbors[nodes_current].reshape((batch_size, q_neighbor_size))
        neighbors_concept = neighbors_concept.reshape(-1)
        neighbors_question = self.concept_neighbors[neighbors_concept].\
            reshape((batch_size, q_neighbor_size * c_neighbor_size)).tolist()
        result = [[] for _ in range(batch_size)]
        for row in range(batch_size):
            key = history_q[row].tolist()
            query = neighbors_question[row]
            for t, k in enumerate(key):
                if k in query:
                    result[row].append(t)
        return result

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

    def get_knowledge_state(self, batch):
        pass