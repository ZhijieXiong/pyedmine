import torch
import torch.nn as nn


from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer


class LBKT(nn.Module, DLSequentialKTModel):
    model_name = "LBKT"

    def __init__(self, params, objects):
        super(LBKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["LBKT"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        dim_question = model_config["embed_config"]["question"]["dim_item"]
        dim_correctness = model_config["embed_config"]["correctness"]["dim_item"]
        dim_h = model_config["dim_h"]

        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.input_layer = nn.Linear(dim_question + dim_correctness, dim_h)
        torch.nn.init.xavier_normal_(self.input_layer.weight)
        self.lbkt_cell = LBKTcell(params, objects)
        self.init_h = nn.Parameter(torch.Tensor(num_concept, dim_h))
        nn.init.xavier_normal_(self.init_h)

    def forward(self, batch):
        question_seq = batch["question_seq"]
        correctness_seq = batch["correctness_seq"]
        time_factor_seq = batch["time_factor_seq"]
        attempt_factor_seq = batch["attempt_factor_seq"]
        hint_factor_seq = batch["hint_factor_seq"]

        batch_size, seq_len = question_seq.size(0), question_seq.size(1)
        question_emb = self.embed_layer.get_emb("question", question_seq)
        correctness_emb = self.embed_layer.get_emb("correctness", correctness_seq)

        correlation_weight = self.objects["LBKT"]["q_matrix"][question_seq]
        acts_emb = torch.relu(self.input_layer(torch.cat([question_emb, correctness_emb], -1)))

        time_factor_seq = time_factor_seq.unsqueeze(-1)
        attempt_factor_seq = attempt_factor_seq.unsqueeze(-1)
        hint_factor_seq = hint_factor_seq.unsqueeze(-1)

        h_init = self.init_h.unsqueeze(0).repeat(batch_size, 1, 1)
        h_pre = h_init
        predict_score = torch.zeros(batch_size, seq_len).to(self.params["device"])
        for t in range(seq_len):
            pred, h = self.lbkt_cell(acts_emb[:, t], correlation_weight[:, t], question_emb[:, t],
                                     time_factor_seq[:, t], attempt_factor_seq[:, t], hint_factor_seq[:, t], h_pre)
            h_pre = h
            predict_score[:, t] = pred

        return predict_score

    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
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


class Layer(nn.Module):
    def __init__(self, dim_h, d, k, b):
        super(Layer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(2 * dim_h, dim_h))
        self.bias = nn.Parameter(torch.zeros(1, dim_h))

        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.bias)

        self.d = d
        self.k = k
        self.b = b

    def forward(self, factor, interact_emb, h):
        gate = self.k + (1 - self.k) / (1 + torch.exp(-self.d * (factor - self.b)))
        w = torch.cat([h, interact_emb], -1).matmul(self.weight) + self.bias
        w = nn.Sigmoid()(w * gate)

        return w


class LBKTcell(nn.Module):
    def __init__(self, params, objects):
        super(LBKTcell, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["LBKT"]
        dim_question = model_config["embed_config"]["question"]["dim_item"]
        r = model_config["r"]
        dim_h = model_config["dim_h"]
        dim_factor = model_config["dim_factor"]
        dropout = model_config["dropout"]
        d = model_config["d"]
        k = model_config["k"]
        b = model_config["b"]

        self.time_gain = Layer(dim_h, d, b, k)
        self.attempt_gain = Layer(dim_h, d, b, k)
        self.hint_gain = Layer(dim_h, d, b, k)

        self.time_weight = nn.Parameter(torch.Tensor(r, dim_h + 1, dim_h))
        nn.init.xavier_normal_(self.time_weight)

        self.attempt_weight = nn.Parameter(torch.Tensor(r, dim_h + 1, dim_h))
        nn.init.xavier_normal_(self.attempt_weight)

        self.hint_weight = nn.Parameter(torch.Tensor(r, dim_h + 1, dim_h))
        nn.init.xavier_normal_(self.hint_weight)

        self.Wf = nn.Parameter(torch.Tensor(1, r))
        nn.init.xavier_normal_(self.Wf)

        self.bias = nn.Parameter(torch.Tensor(1, dim_h))
        nn.init.xavier_normal_(self.bias)

        self.gate3 = nn.Linear(2 * dim_h + 3 * dim_factor, dim_h)
        torch.nn.init.xavier_normal_(self.gate3.weight)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_question + dim_h, dim_h)
        torch.nn.init.xavier_normal_(self.output_layer.weight)
        self.sig = nn.Sigmoid()

    def forward(self, interact_emb, correlation_weight, topic_emb, time_factor, attempt_factor, hint_factor, h_pre):
        model_config = self.params["models_config"]["LBKT"]
        dim_h = model_config["dim_h"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        dim_factor = model_config["dim_factor"]

        # bs *1 * memory_size , bs * memory_size * d_k
        h_pre_tilde = torch.squeeze(torch.bmm(correlation_weight.unsqueeze(1), h_pre), 1)
        # predict performance
        predict_score = torch.sum(self.sig(self.output_layer(torch.cat([h_pre_tilde, topic_emb], -1))), -1) / dim_h

        # characterize each behavior's effect
        time_gain = self.time_gain(time_factor, interact_emb, h_pre_tilde)
        attempt_gain = self.attempt_gain(attempt_factor, interact_emb, h_pre_tilde)
        hint_gain = self.hint_gain(hint_factor, interact_emb, h_pre_tilde)

        # capture the dependency among different behaviors
        pad = torch.ones_like(time_factor)  # bs * 1
        time_gain1 = torch.cat([time_gain, pad], -1)  # bs * num_units + 1
        attempt_gain1 = torch.cat([attempt_gain, pad], -1)
        hint_gain1 = torch.cat([hint_gain, pad], -1)
        # bs * r  *num_units: bs * num_units + 1 ,r * num_units + 1 *num_units
        fusion_time = torch.matmul(time_gain1, self.time_weight)
        fusion_attempt = torch.matmul(attempt_gain1, self.attempt_weight)
        fusion_hint = torch.matmul(hint_gain1, self.hint_weight)
        fusion_all = fusion_time * fusion_attempt * fusion_hint
        # 1 * r, bs * r * num_units -> bs * 1 * num_units -> bs * num_units
        fusion_all = torch.matmul(self.Wf, fusion_all.permute(1, 0, 2)).squeeze(1) + self.bias
        learning_gain = torch.relu(fusion_all)

        LG = torch.matmul(correlation_weight.unsqueeze(-1), learning_gain.unsqueeze(1))

        # forget effect
        forget_gate = self.gate3(torch.cat([h_pre, interact_emb.unsqueeze(1).repeat(1, num_concept, 1),
                                            time_factor.unsqueeze(1).repeat(1, num_concept, dim_factor),
                                            attempt_factor.unsqueeze(1).repeat(1, num_concept, dim_factor),
                                            hint_factor.unsqueeze(1).repeat(1, num_concept, dim_factor)], -1))
        LG = self.dropout(LG)
        h = h_pre * self.sig(forget_gate) + LG

        return predict_score, h
