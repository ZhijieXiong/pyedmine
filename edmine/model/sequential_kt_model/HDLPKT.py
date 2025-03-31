import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer
from edmine.model.loss import binary_cross_entropy


class HDLPKT(nn.Module, DLSequentialKTModel):
    model_name = "HDLPKT"
    def __init__(self, params, objects):
        super(HDLPKT, self).__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["HDLPKT"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        dim_k = model_config["dim_k"]
        dim_a = model_config["dim_a"]
        dim_e = model_config["dim_e"]
        dropout = model_config["dropout"]
        max_seq_length = model_config["max_seq_length"]
        
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        
        self.linear_1 = nn.Linear(dim_a + dim_e + dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(4 * dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(4 * dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(3 * dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(dim_e + dim_k, dim_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.loss_function = nn.BCELoss()
        self.emb_dropout = nn.Dropout(0.3)
        self.dropout1 = nn.Dropout(dropout)
        self.attention_linear1 = nn.Linear(dim_k, dim_k)
        self.attention_linear2 = nn.Linear(2 * dim_k, dim_k)
        self.attention_linear3 = nn.Linear(2*dim_k, 2)
        self.rnn1 = nn.LSTM(
            input_size=dim_k * 2,
            hidden_size=dim_k * 2,
            bidirectional=True,
            batch_first=True
        )
        self.seq_level_mlp = nn.Sequential(
            nn.Linear(dim_k * 2, 2, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(max_seq_length, max_seq_length, (1, 2))
        self.vae = VAE(dim_k * 2, dim_k, dim_k).to(self.params["device"])
        self.deno=nn.Linear(num_concept, 2)
        
    def get_predict_score(self, batch, seq_start=2):
        model_config = self.params["models_config"]["HDLPKT"]
        dim_k = model_config["dim_k"]
        dim_a = model_config["dim_a"]
        max_seq_length = model_config["max_seq_length"]
        q_matrix = self.objects["HDLPKT"]["q_matrix"]
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        
        user_id = batch["user_id"]
        exercise_seq_tensor = batch["question_seq"]
        skill_seq_tensor = batch["concept_seq"]
        correct_seq_tensor = batch["correctness_seq"]
        answertime_seq_tensor = batch["use_time_seq"]
        intervaltime_seq_tensor = batch["interval_time_seq"]
        seqs_length = batch["seq_len"]
        mask_labels = batch["mask_seq"]
        mask = batch["mask_seq"].float()
        
        batch_size = exercise_seq_tensor.size()[0]
        stu_emb = self.embed_layer.get_emb("user", user_id)
        exer_emb = self.embed_layer.get_emb("question", exercise_seq_tensor)
        knowledge_emb = self.embed_layer.get_emb("interaction", skill_seq_tensor + num_concept * mask_labels)
        
        k_ = torch.cat((exer_emb, knowledge_emb), dim=2)
        stu_deno_singal, _ = self.target_exer_discriminator(stu_emb, k_, mask)
        _, seq_level_score = self.seq_level_vae(k_, mask)
        seq_level_score = seq_level_score[:, :, 1]
        deno_siginal = (1 - (seq_level_score * stu_deno_singal))
        deno_siginal = deno_siginal.unsqueeze(2).expand(-1, -1, dim_k)
        deno_siginal = deno_siginal.masked_fill(deno_siginal == -np.inf, 0)
        answertime_embedding = self.embed_layer.get_emb("use_time", answertime_seq_tensor) * deno_siginal
        intervaltime_embedding = self.embed_layer.get_emb("interval_time", intervaltime_seq_tensor) * deno_siginal
        exercise_embeding = self.embed_layer.get_emb("question", exercise_seq_tensor) * deno_siginal
        a_data = correct_seq_tensor.view(-1, 1).repeat(1, dim_a).view(batch_size, -1, dim_a)
        all_learning = self.linear_1(torch.cat((exercise_embeding, answertime_embedding, a_data), 2))
        
        h_pre = nn.init.xavier_uniform_(torch.zeros(num_concept, dim_k)).repeat(batch_size, 1, 1).to(
            self.params["device"])
        h_pre = torch.as_tensor(h_pre, dtype=torch.float)
        h_tilde_pre = None
        learning_pre = torch.zeros(batch_size, dim_k).to(self.params["device"])
        pred = torch.zeros(batch_size, max_seq_length).to(self.params["device"])
        for t in range(max(seqs_length) - 1):
            e = exercise_seq_tensor[:, t]
            q_e = q_matrix[e].view(batch_size, 1, -1)
            it = intervaltime_embedding[:, t]
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, dim_k)
            learning = all_learning[:, t]
            IG = self.linear_2(torch.cat((learning_pre, it, learning, h_tilde_pre), 1))
            IG = self.tanh(IG)
            Gamma_l = self.sig(self.linear_3(torch.cat((learning_pre, it, learning, h_tilde_pre), 1)))
            LG = Gamma_l * ((IG + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            n_skill = LG_tilde.size(1)
            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG.repeat(1, n_skill).view(batch_size, -1, dim_k),
                it.repeat(1, n_skill).view(batch_size, -1, dim_k)
            ), 2)))
            h = LG_tilde + gamma_f * h_pre
            h_tilde = q_matrix[exercise_seq_tensor[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, dim_k)
            y = self.sig(self.linear_5(torch.cat((exercise_embeding[:, t + 1], h_tilde), 1))).sum(1) / dim_k
            pred[:, t + 1] = y
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        mask_seq = torch.ne(batch["mask_seq"], 0)
        return {
            "predict_score_batch": pred[:, 1:],
            "predict_score": torch.masked_select(pred[:, seq_start-1:], mask_seq[:, seq_start-1:])
        }

    def get_predict_loss(self, batch, seq_start=2):        
        num_concept = self.objects["dataset"]["q_table"].shape[1]
        
        exercise_seq_tensor = batch["question_seq"]
        skill_seq_tensor = batch["concept_seq"]
        correctness_seq = batch["correctness_seq"]
        mask = batch["mask_seq"].float()
        
        exer_emb = self.embed_layer.get_emb("question", exercise_seq_tensor)
        knowledge_emb = self.embed_layer.get_emb("interaction", skill_seq_tensor + num_concept * correctness_seq)
        k_ = torch.cat((exer_emb, knowledge_emb), dim=2)
        rec_loss, _ = self.seq_level_vae(k_, mask)

        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score_result = self.get_predict_score(batch, seq_start)
        predict_score = predict_score_result["predict_score"]
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        return {
            "total_loss": predict_loss + torch.sum(rec_loss, dim=0),
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "rec_loss loss": {
                    "value": rec_loss.detach().cpu().item(),
                    "num_sample": 1
                },
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_result["predict_score_batch"]
        }

    def target_exer_discriminator(self, q, k, mask):
        model_config = self.params["models_config"]["HDLPKT"]
        dim_k = model_config["dim_k"]
        
        mask1 = mask.unsqueeze(2).expand(-1, -1, 2 * dim_k)
        item_seq_emb = self.emb_dropout(k) * mask1
        encoder_item_seq_emb_bi_direction, _ = self.rnn1(item_seq_emb)

        rnn1_hidden = int(encoder_item_seq_emb_bi_direction.shape[-1] / 2)
        encoder_item_seq_emb = encoder_item_seq_emb_bi_direction[:, :, :rnn1_hidden] + \
                               encoder_item_seq_emb_bi_direction[:, :, rnn1_hidden:]

        q = q.unsqueeze(1).expand(-1, k.size(1), -1)
        q_ = self.dropout1(self.attention_linear1(q))
        k_1 = self.dropout1(self.attention_linear2(encoder_item_seq_emb))
        k_2 = self.dropout1(self.attention_linear2(k))

        cl_loss = 0
        k_=torch.cat((k_1,k_2),dim=2)
        q_=torch.repeat_interleave(q_,2,dim=2)
        alpha = torch.sigmoid(self.attention_linear3(torch.tanh(q_ + k_)))
        gumbel_softmax_alpha = F.gumbel_softmax(alpha, tau=100, hard=True)
        mask = mask.unsqueeze(2).expand(-1, -1, 2)
        gumbel_softmax_alpha = gumbel_softmax_alpha.masked_fill(mask == 0, -np.inf)

        return gumbel_softmax_alpha[:, :, 1], cl_loss

    def seq_level_vae(self, item_seq, mask):
        model_config = self.params["models_config"]["HDLPKT"]
        dim_k = model_config["dim_k"]
        
        mask1 = mask.unsqueeze(2).expand(-1, -1, 2 * dim_k)
        item_seq_emb = self.emb_dropout(item_seq) * mask1
        encoder_item_seq_emb_bi_direction, _ = self.rnn1(item_seq_emb)
        rnn1_hidden = int(encoder_item_seq_emb_bi_direction.shape[-1] / 2)
        encoder_item_seq_emb = encoder_item_seq_emb_bi_direction[:, :, :rnn1_hidden] + \
                               encoder_item_seq_emb_bi_direction[:, :, rnn1_hidden:]

        x_reconst, mu, log_var = self.vae(encoder_item_seq_emb)
        reconst_loss = F.mse_loss(x_reconst * mask1, encoder_item_seq_emb * mask1, reduction='sum')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        element_wise_reconstruction_loss = reconst_loss + kl_div
        concat_shuffled_and_origin = torch.stack((x_reconst, encoder_item_seq_emb), dim=-1)
        concat_shuffled_and_origin = self.conv(concat_shuffled_and_origin)
        concat_shuffled_and_origin = torch.squeeze(concat_shuffled_and_origin)
        concat_shuffled_and_origin = self.emb_dropout(concat_shuffled_and_origin)
        concat_shuffled_and_origin = nn.ReLU(inplace=True)(concat_shuffled_and_origin)
        reconstruct_score = self.seq_level_mlp(concat_shuffled_and_origin).squeeze()
        mask2 = mask.unsqueeze(2).expand(-1, -1, 2)
        reconstruct_score = reconstruct_score * mask2
        gumbel_softmax_reconstruct_score = F.gumbel_softmax(reconstruct_score, tau=100, hard=True)
        gumbel_softmax_reconstruct_score = gumbel_softmax_reconstruct_score.masked_fill(mask2 == 0, -np.inf)

        return element_wise_reconstruction_loss, gumbel_softmax_reconstruct_score
    
    def get_predict_score_on_target_question(self, batch, target_index, target_question):
        pass

    def get_knowledge_state(self, batch):
        pass


class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var