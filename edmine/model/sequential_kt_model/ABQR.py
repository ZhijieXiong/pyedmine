import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.loss import binary_cross_entropy


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x


class MLP_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)


class BGRL(nn.Module):
    def __init__(self, dim_emb, dropout, gcn_adj):
        super(BGRL, self).__init__()
        self.gcn_adj = gcn_adj
    
        self.online_encoder = GCNConv(dim_emb, dim_emb, dropout)
        self.decoder = GCNConv(dim_emb, dim_emb, dropout)
        self.predictor = MLP_Predictor(dim_emb, dim_emb, dim_emb)
        self.target_encoder = copy.deepcopy(self.online_encoder)

        self.fc1 = nn.Linear(dim_emb, dim_emb)
        self.fc2 = nn.Linear(dim_emb, dim_emb)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.enc_mask_token = nn.Parameter(torch.zeros(1, dim_emb))
        self.encoder_to_decoder = nn.Linear(dim_emb, dim_emb, bias=False)

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        num_noise_nodes = int(0.1 * num_mask_nodes)
        perm_mask = torch.randperm(num_mask_nodes, device=x.device)
        token_nodes = mask_nodes[perm_mask[: int(0.9 * num_mask_nodes)]]
        noise_nodes = mask_nodes[perm_mask[-int(0.1 * num_mask_nodes):]]
        noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

        out_x = x.clone()
        out_x[token_nodes] = 0.0
        out_x[noise_nodes] = x[noise_to_be_chosen]

        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes, keep_nodes

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def forward(self, x, adj, perb=None):
        if perb is None:
            return x + self.online_encoder(x, self.gcn_adj), 0

        x1, adj1 = x, copy.deepcopy(adj)
        x2, adj2 = x + perb, copy.deepcopy(adj)

        embed = x2 + self.online_encoder(x2, adj2)

        online_x = self.online_encoder(x1, adj1)
        online_y = self.online_encoder(x2, adj2)

        with torch.no_grad():
            target_y = self.target_encoder(x1, adj1).detach()
            target_x = self.target_encoder(x2, adj2).detach()

        online_x = self.predictor(online_x)
        online_y = self.predictor(online_y)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean()

        return embed, loss


class ABQR(nn.Module, DLSequentialKTModel):
    model_name = "ABQR"
    
    def __init__(self, params, objects):
        super(ABQR, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["ABQR"]
        num_question = self.objects["dataset"]["q_table"].shape[0]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        dim_emb = model_config["dim_emb"]
        dropout = model_config["dropout"]
        gcn_adj = self.objects["ABQR"]["gcn_adj"]
        
        self.gcl = BGRL(dim_emb, dropout, gcn_adj)
        self.gcn = GCNConv(dim_emb, dim_emb, dropout)
        self.pro_embed = nn.Parameter(torch.ones((num_question, dim_emb)))
        nn.init.xavier_uniform_(self.pro_embed)
        self.ans_embed = nn.Embedding(2, dim_emb)

        self.attn = nn.MultiheadAttention(dim_emb, 8, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_layer_norm = nn.LayerNorm(dim_emb)

        self.FFN = nn.Sequential(
            nn.Linear(dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_emb, dim_emb),
            nn.Dropout(dropout),
        )
        self.FFN_layer_norm = nn.LayerNorm(dim_emb)
        self.pred = nn.Linear(dim_emb, 1)
        self.lstm = nn.LSTM(dim_emb, dim_emb, batch_first=True)
        self.origin_lstm = nn.LSTM(2 * dim_emb, 2 * dim_emb, batch_first=True)
        self.oppo_lstm = nn.LSTM(dim_emb, dim_emb, batch_first=True)
        self.origin_lstm2 = nn.LSTM(dim_emb, dim_emb, batch_first=True)
        self.oppo_lstm2 = nn.LSTM(dim_emb, dim_emb, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.origin_out = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_emb, 1)
        )
        self.oppo_out = nn.Sequential(
            nn.Linear(2 * dim_emb, dim_emb),
            nn.ReLU(),
            nn.Linear(dim_emb, 1)
        )

        self.encoder_lstm = nn.LSTM(dim_emb, dim_emb, batch_first=True)
        self.decoder_lstm = nn.LSTM(dim_emb, dim_emb, batch_first=True)

        self.enc_token = nn.Parameter(torch.rand(1, dim_emb))
        self.enc_dec = nn.Linear(dim_emb, dim_emb)

        self.classify = nn.Sequential(
            nn.Linear(dim_emb, num_concept)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                
    def get_predict_score(self, batch, seq_start=2):
        last_pro = batch["question_seq"][:, :-1]
        next_pro = batch["question_seq"][:, 1:]
        last_ans = batch["correctness_seq"][:, :-1]
        gcn_adj = self.objects["ABQR"]["gcn_adj"]
        
        pro_embed, _ = self.gcl(self.pro_embed, gcn_adj, None)
        last_pro_embed = F.embedding(last_pro, pro_embed)
        next_pro_embed = F.embedding(next_pro, pro_embed)
        ans_embed = self.ans_embed(last_ans)

        X = last_pro_embed + ans_embed
        X = self.dropout(X)
        X, _ = self.lstm(X)

        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_batch = torch.sigmoid(self.origin_out(torch.cat([X, next_pro_embed], dim=-1))).squeeze(-1)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }
        

    def get_predict_loss(self, batch, seq_start=2):
        last_pro = batch["question_seq"][:, :-1]
        last_ans = batch["correctness_seq"][:, :-1]
        model_config = self.params["models_config"]["ABQR"]
        dim_emb = model_config["dim_emb"]
        num_question = self.objects["dataset"]["q_table"].shape[0]
        gcn_adj = self.objects["ABQR"]["gcn_adj"]
        trainer_config = self.params["trainer_config"]
        model_name = trainer_config["model_name"]
        optimizer = self.objects["optimizers"][model_name]
        
        perturb_shape = (num_question, dim_emb)
        step_size = 3e-2
        step_m = 3
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(self.params["device"])
        perturb.requires_grad_()
        
        pro_embed, contrast_loss = self.gcl(self.pro_embed, gcn_adj, perturb)
        contrast_loss = 0.1 * contrast_loss
        last_pro_embed = F.embedding(last_pro, pro_embed)
        ans_embed = self.ans_embed(last_ans)
        X = last_pro_embed + ans_embed
        X = self.dropout(X)
        X, _ = self.lstm(X)
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        
        loss = predict_loss + contrast_loss
        loss /= step_m
        optimizer.zero_grad()
        for _ in range(step_m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            pro_embed, contrast_loss = self.gcl(self.pro_embed, gcn_adj, perturb)
            contrast_loss = 0.1 * contrast_loss
            last_pro_embed = F.embedding(last_pro, pro_embed)
            ans_embed = self.ans_embed(last_ans)
            X = last_pro_embed + ans_embed
            X = self.dropout(X)
            X, _ = self.lstm(X)
            mask_seq = torch.ne(batch["mask_seq"], 0)
            predict_score_result = self.get_predict_score(batch)
            predict_score = predict_score_result["predict_score"]
            ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
            predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
            
            loss = predict_loss + contrast_loss
            loss /= step_m
        
        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "gcl loss": {
                    "value": contrast_loss.detach().cpu().item() * num_question,
                    "num_sample": num_question
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def get_knowledge_state(self, batch):
        pass
    