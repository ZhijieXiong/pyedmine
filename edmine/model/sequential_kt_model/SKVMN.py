import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear, Dropout
from torch.nn.init import kaiming_normal_

from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel


class DKVMNHeadGroup(nn.Module):
    def __init__(self, size_memory, dim_kv, is_write, device):
        super(DKVMNHeadGroup, self).__init__()
        self.size_memory = size_memory
        self.dim_kv = dim_kv
        self.is_write = is_write
        self.device = device
        if self.is_write:
            self.erase = torch.nn.Linear(dim_kv, dim_kv)
            self.add = torch.nn.Linear(dim_kv, dim_kv)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    @staticmethod
    def addressing(control_input, memory):
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = F.softmax(similarity_score, dim=1)
        return correlation_weight

    def read(self, memory, read_weight):
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.dim_kv)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.size_memory, self.dim_kv)
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, control_input, memory, write_weight=None):
        if write_weight is None:
            write_weight = self.addressing(control_input, memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.dim_kv)
        add_reshape = add_signal.view(-1, 1, self.dim_kv)
        write_weight_reshape = write_weight.view(-1, self.size_memory, 1)
        erase_mul = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        memory = memory.to(self.device)
        if add_mul.shape[0] < memory.shape[0]:
            sub_memory = memory[:add_mul.shape[0],:,:]
            new_memory = torch.cat([sub_memory * (1 - erase_mul) + add_mul, memory[add_mul.shape[0]:,:,:]], dim=0)
        else:
            new_memory = memory * (1 - erase_mul) + add_mul
        return new_memory


class DKVMN(nn.Module):
    def __init__(self, size_memory, dim_kv, memory_key, device):
        super(DKVMN, self).__init__()
        self.size_memory = size_memory
        self.key_head = DKVMNHeadGroup(size_memory, dim_kv, False, device)
        self.value_head = DKVMNHeadGroup(size_memory, dim_kv, True, device)
        self.memory_key = memory_key

    def attention(self, control_input):
        return self.key_head.addressing(control_input=control_input, memory=self.memory_key)

    def read(self, read_weight, memory_value):
        return self.value_head.read(memory=memory_value, read_weight=read_weight)

    def write(self, write_weight, control_input, memory_value):
        return self.value_head.write(control_input=control_input, memory=memory_value, write_weight=write_weight)


class SKVMN(Module, DLSequentialKTModel):
    model_name = "SKVMN"
    
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        model_config = params["models_config"]["SKVMN"]
        dim_kv = model_config["dim_kv"]
        size_memory = model_config["size_memory"]
        dropout = model_config["dropout"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.Mk = Parameter(torch.Tensor(size_memory, dim_kv))
        self.Mv0 = Parameter(torch.Tensor(size_memory, dim_kv)) 
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.mem = DKVMN(size_memory, dim_kv, self.Mk, params["device"])
        self.a_embed = nn.Linear(dim_kv * 2, dim_kv, bias=True)
        self.f_layer = Linear(dim_kv * 2, dim_kv)
        self.hx = Parameter(torch.Tensor(1, dim_kv))
        self.cx = Parameter(torch.Tensor(1, dim_kv))
        kaiming_normal_(self.hx)
        kaiming_normal_(self.cx)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(dim_kv, 1)
        self.lstm_cell = nn.LSTMCell(dim_kv, dim_kv)

    def ut_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(dtype=torch.bool)

    def triangular_layer(self, correlation_weight, batch_size, seq_len):
        model_config = self.params["models_config"]["SKVMN"]
        a = model_config["a"]
        b = model_config["b"]
        c = model_config["c"]
        batch_identity_indices = []

        # w'= max((w-a)/(b-a), (c-w)/(c-b))
        # min(w', 0)
        correlation_weight = correlation_weight.view(batch_size * seq_len, -1)
        correlation_weight = torch.cat([correlation_weight[i] for i in range(correlation_weight.shape[0])], 0).unsqueeze(0)
        correlation_weight = torch.cat([(correlation_weight-a)/(b-a), (c-correlation_weight)/(c-b)], 0)
        correlation_weight, _ = torch.min(correlation_weight, 0)
        w0 = torch.zeros(correlation_weight.shape[0]).to(self.params["device"])
        correlation_weight = torch.cat([correlation_weight.unsqueeze(0), w0.unsqueeze(0)], 0)
        correlation_weight, _ = torch.max(correlation_weight, 0)

        identity_vector_batch = torch.zeros(correlation_weight.shape[0]).to(self.params["device"])

        # >=0.6的值置2，0.1-0.6的值置1，0.1以下的值置0
        # mask = correlation_weight.lt(0.1)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.lt(0.1), 0)
        # mask = correlation_weight.ge(0.1)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.1), 1)
        # mask = correlation_weight.ge(0.6)
        _identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.6), 2)

        identity_vector_batch = _identity_vector_batch.view(batch_size * seq_len, -1)
        identity_vector_batch = torch.reshape(identity_vector_batch,[batch_size, seq_len, -1]) #输出u(x) [batch_size, seqlen, size_memory]

        # A^2
        iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2), dim=2, keepdim=True)
        iv_square_norm = iv_square_norm.repeat((1, 1, iv_square_norm.shape[1]))
        # B^2.T
        unique_iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2), dim=2, keepdim=True)
        unique_iv_square_norm = unique_iv_square_norm.repeat((1, 1, seq_len)).transpose(2, 1)
        # A * B.T
        iv_matrix_product = torch.bmm(identity_vector_batch, identity_vector_batch.transpose(2,1)) # A * A.T 
        # A^2 + B^2 - 2A*B.T
        iv_distances = iv_square_norm + unique_iv_square_norm - 2 * iv_matrix_product
        iv_distances = torch.where(iv_distances>0.0, torch.tensor(-1e32).to(self.params["device"]), iv_distances) #求每个batch内时间步t与t-lambda的相似距离（如果identity_vector一样，距离为0）
        masks = self.ut_mask(iv_distances.shape[1]).to(self.params["device"])
        mask_iv_distances = iv_distances.masked_fill(masks, value=torch.tensor(-1e32).to(self.params["device"])) #当前时刻t以前相似距离为0的依旧为0，其他为mask（即只看对角线以前）
        idx_matrix = torch.arange(0,seq_len * seq_len,1).reshape(seq_len,-1).repeat(batch_size,1,1).to(self.params["device"])
        final_iv_distance = mask_iv_distances + idx_matrix 
        values, indices = torch.topk(final_iv_distance, 1, dim=2, largest=True) #防止t以前存在多个相似距离为0的,因此加上idx取距离它最近的t - lambda
        
        _values = values.permute(1,0,2)
        _indices = indices.permute(1,0,2)
        batch_identity_indices = (_values >= 0).nonzero() #找到t
        identity_idx = []
        for identity_indices in batch_identity_indices:
            pre_idx = _indices[identity_indices[0],identity_indices[1]] #找到t-lamda
            idx = torch.cat([identity_indices[:-1],pre_idx], dim=-1)
            identity_idx.append(idx)
        if len(identity_idx) > 0:
            identity_idx = torch.stack(identity_idx, dim=0)
        else:
            identity_idx = torch.tensor([]).to(self.params["device"])

        return identity_idx 


    def forward(self, batch):
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        interaction_seq = num_concept * batch["correctness_seq"].unsqueeze(-1)
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        batch_size, seq_len = batch["correctness_seq"].shape[0], batch["correctness_seq"].shape[1]    
                  
        concept_emb = self.embed_layer.get_emb_fused1(
            "concept", q2c_transfer_table, q2c_mask_table, batch["question_seq"]
        )
        value_read_content_l = []
        input_embed_l = []
        correlation_weight_list = []
        ft = []

        #每个时间步计算一次attn，更新memory key & memory value
        mem_value = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1).to(self.params["device"])
        for i in range(seq_len):
            # Attention
            concept_emb_i = concept_emb.permute(1,0,2)[i]
            correlation_weight = self.mem.attention(concept_emb_i).to(self.params["device"])
            # Read Process
            read_content = self.mem.read(correlation_weight, mem_value)
            # modify
            correlation_weight_list.append(correlation_weight)
            # save intermedium data
            value_read_content_l.append(read_content)
            input_embed_l.append(concept_emb_i)
            batch_predict_input = torch.cat([read_content, concept_emb_i], 1)
            f = torch.tanh(self.f_layer(batch_predict_input))
            ft.append(f)
            
            y = self.embed_layer.get_emb_fused1(
                "interaction", q2c_transfer_table, q2c_mask_table, 
                batch["question_seq"][:,i], other_item_index=interaction_seq[:,i]
            )
            write_embed = torch.cat([f, y], 1)
            write_embed = self.a_embed(write_embed).to(self.params["device"])
            new_memory_value = self.mem.write(correlation_weight, write_embed, mem_value)
            mem_value = new_memory_value

        w = torch.cat([correlation_weight_list[i].unsqueeze(1) for i in range(seq_len)], 1)
        ft = torch.stack(ft, dim=0)
        idx_values = self.triangular_layer(w, batch_size, seq_len)

        # Hop-LSTM
        hidden_state, cell_state = [], []
        hx, cx = self.hx.repeat(batch_size, 1), self.cx.repeat(batch_size, 1)
        for i in range(seq_len): # 逐个ex进行计算
            for j in range(batch_size):
                if idx_values.shape[0] != 0 and i == idx_values[0][0] and j == idx_values[0][1]:
                    # e.g 在t=3时，第2个序列的hidden应该用t=1时的hidden,同理cell_state
                    hx[j,:] = hidden_state[idx_values[0][2]][j]
                    cx = cx.clone()
                    cx[j,:] = cell_state[idx_values[0][2]][j]
                    idx_values = idx_values[1:]
            hx, cx = self.lstm_cell(ft[i], (hx, cx))
            hidden_state.append(hx)
            cell_state.append(cx)
        hidden_state = torch.stack(hidden_state, dim=0).permute(1,0,2)
        cell_state = torch.stack(cell_state, dim=0).permute(1,0,2)
        p = self.p_layer(self.dropout_layer(hidden_state))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)
        return p

        # 时间优化
        # copy_ft = torch.repeat_interleave(ft, repeats=seq_len, dim=0).reshape(batch_size, seq_len, seq_len, -1)
        # mask = torch.tensor(np.eye(seq_len, seq_len)).to(self.params["device"])
        # copy_mask = mask.repeat(batch_size,1,1)
        # for i in range(idx_values.shape[0]):
        #     n = idx_values[i][1]
        #     t = idx_values[i][0]
        #     t_a = idx_values[i][2]
        #     copy_ft[n][t][t-t_a] = copy_ft[n][t][t]
        #     if t_a + 1 != t:
        #         copy_mask[n][t][t_a+1] = 1
        #         copy_mask[n][t][t] = 0
        # copy_ft_reshape = torch.reshape(copy_ft,(batch_size, seq_len * seq_len,-1))
        # p = self.p_layer(self.dropout_layer(copy_ft_reshape))
        # p = torch.sigmoid(p)
        # p = torch.reshape(p.squeeze(-1),(batch_size, -1))
        # copy_mask_reshape = torch.reshape(copy_mask, (batch_size,-1))
        # copy_mask_reshape = copy_mask_reshape.ge(1)
        # p = torch.masked_select(p, copy_mask_reshape).reshape(batch_size,-1)
        # return p
        
    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(batch_size, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
        # 如此设定是为了做cold start evaluation
        predict_score_batch = self.forward(batch)[:, 1:]
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass