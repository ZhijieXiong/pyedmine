import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.loss import binary_cross_entropy


def positive_activate(mode, input_tensor):
    if mode == 'sigmoid':
        return torch.sigmoid(input_tensor)
    if mode == 'softplus':
        return F.softplus(input_tensor)
    if mode == 'relu':
        return torch.relu(input_tensor)
    if mode == 'softmax':
        return input_tensor.softmax(0)
    if mode == 'none':
        return input_tensor
    
    
class Positive_Linear(nn.Module):
    def __init__(self, d_in, d_out, mode):
        super(Positive_Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out))
        self.mode = mode
    
    def forward(self, input_tensor):
        return input_tensor.matmul(positive_activate(self.mode, self.weight))


class GRKT(nn.Module, DLSequentialKTModel):
    model_name = "GRKT"
    
    def __init__(self, params, objects):
        super(GRKT, self).__init__()
        self.params = params
        self.objects = objects
        
        # num_concept, num_question, dim_hidden, k_hidden, pos_mode, k_hop, rel_map, thresh, pre_map, alpha, tau
        model_config = self.params["models_config"]["GRKT"]
        num_concept = model_config["num_concept"]
        num_question = model_config["num_question"]
        dim_hidden = model_config["dim_hidden"]
        k_hidden = model_config["k_hidden"]
        pos_mode = model_config["pos_mode"]
        k_hop = model_config["k_hop"]
        thresh = model_config["thresh"]
        rel_map = objects["GRKT"]["rel_map"]
        pre_map = objects["GRKT"]["pre_map"]
        
        self.know_embedding = nn.Embedding(num_concept + 1, dim_hidden, padding_idx = 0)
        self.prob_embedding = nn.Embedding(num_question + 1, dim_hidden, padding_idx = 0)
  
        pos_mode = pos_mode
        self.init_hidden = nn.Parameter(torch.randn(num_concept + 1, k_hidden))
        self.know_master_proj = Positive_Linear(k_hidden, 1, 'softmax')
        
        self.req_matrix = nn.Linear(dim_hidden, dim_hidden, bias = False)
        self.rel_matrix = nn.Linear(dim_hidden, dim_hidden, bias = False)

        self.agg_rel_matrix = nn.ModuleList([Positive_Linear(
            k_hidden, k_hidden, 'softmax') for _ in range(k_hop)])
        self.agg_pre_matrix = nn.ModuleList([Positive_Linear(
            k_hidden, k_hidden, 'softmax') for _ in range(k_hop)])
        self.agg_sub_matrix = nn.ModuleList([Positive_Linear(
            k_hidden, k_hidden, 'softmax') for _ in range(k_hop)])

        self.prob_diff_mlp = nn.Sequential(
            nn.Linear(2*dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )

        self.gain_ffn = nn.Sequential(
            nn.Linear(2*dim_hidden + k_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, k_hidden),
        )

        self.gain_matrix_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.gain_matrix_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.gain_matrix_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.gain_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.gain_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.gain_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])

        self.loss_ffn = nn.Sequential(
            nn.Linear(2*dim_hidden + k_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, k_hidden),
        )

        self.loss_matrix_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.loss_matrix_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.loss_matrix_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.loss_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.loss_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.loss_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
		
        rel_map = (rel_map > thresh).astype(np.float32)
        self.rel_map = torch.BoolTensor(rel_map).to(self.params["device"])	# [NK, NK]
        pre_map = (pre_map > thresh).astype(np.float32)
        self.pre_map = torch.BoolTensor(pre_map).to(self.params["device"])	# [NK, NK]

        if thresh == 0:
            self.rel_map = torch.ones_like(self.rel_map)	# [NK, NK]
            self.pre_map = torch.ones_like(self.pre_map)	# [NK, NK]

        self.sub_map = self.pre_map.transpose(-1, -2)

        self.decision_mlp = nn.Sequential(
            nn.Linear(4*dim_hidden + k_hidden, 2*dim_hidden),
            nn.ReLU(),
            nn.Linear(2*dim_hidden, 2),
        )

        self.learn_mlp = nn.Sequential(
            nn.Linear(4*dim_hidden + k_hidden, 2*dim_hidden),
            nn.ReLU(),
            nn.Linear(2*dim_hidden, k_hidden),
        )

        self.learn_matrix_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_matrix_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_matrix_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])

        self.learn_kernel_matrix_rel = nn.ModuleList(
            [nn.Linear(dim_hidden, k_hidden, bias = False)] + \
                [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop - 1)])
        self.learn_kernel_matrix_pre = nn.ModuleList(
            [nn.Linear(dim_hidden, k_hidden, bias = False)] + \
                [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop - 1)])
        self.learn_kernel_matrix_sub = nn.ModuleList(
            [nn.Linear(dim_hidden, k_hidden, bias = False)] + \
                [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop - 1)])
        self.learn_kernel_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_kernel_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.learn_kernel_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])

        self.forget_kernel_matrix_rel = nn.ModuleList(
            [nn.Linear(dim_hidden, k_hidden, bias = False)] + \
                [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop - 1)])
        self.forget_kernel_matrix_pre = nn.ModuleList(
            [nn.Linear(dim_hidden, k_hidden, bias = False)] + \
                [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop - 1)])
        self.forget_kernel_matrix_sub = nn.ModuleList(
            [nn.Linear(dim_hidden, k_hidden, bias = False)] + \
                [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop - 1)])
        self.forget_kernel_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.forget_kernel_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])
        self.forget_kernel_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias = False) for _ in range(k_hop)])

    def forward(self, batch):
        model_config = self.params["models_config"]["GRKT"]
        num_concept = model_config["num_concept"]
        dim_hidden = model_config["dim_hidden"]
        k_hidden = model_config["k_hidden"]
        k_hop = model_config["k_hop"]
        alpha = model_config["alpha"]
        tau = model_config["tau"]
        q_table = self.objects["dataset"]["q_table_tensor"]
        
        # knows (bs, seq_len, K) int
        # corrs (bs, seq_len) bool
        # probs (bs, seq_len) int
        # times (bs, seq_len) int
        # K 数据集中一道习题对应知识点数目最大值
        corrs = torch.ne(batch["correctness_seq"], 0)
        probs = batch['question_seq']
        times = batch['time_seq']
        knows = q_table[probs]

        B, S, K = knows.size()
        DH, KH, NK = dim_hidden, k_hidden, num_concept + 1

        h = self.init_hidden.repeat(B, 1, 1)						# [B, NK, KH]
        h_initial = h												# [B, NK, KH]

        total_know_embedding = self.know_embedding.weight			# [NK, DH]
        prob_embedding = self.prob_embedding(probs) 				# [B, S, DH]
        know_embedding = self.know_embedding(knows) 				# [B, S, K, DH]
        alpha_matrix = self.req_matrix(prob_embedding).matmul(
            total_know_embedding.transpose(-1, -2)).sigmoid()		# [B, S, NK]

        beta_matrix = self.req_matrix(total_know_embedding).matmul(
            total_know_embedding.transpose(-1, -2)).sigmoid()		# [NK, NK]

        rel_map = self.rel_map
        pre_map = self.pre_map
        sub_map = self.sub_map

        beta_rel_tilde = beta_matrix*rel_map/(rel_map.sum(-1, True).clamp(1) + 1e-8)	# [NK, NK]
        beta_pre_tilde = beta_matrix*pre_map/(pre_map.sum(-1, True).clamp(1) + 1e-8)	# [NK, NK]
        beta_sub_tilde = beta_matrix*sub_map/(sub_map.sum(-1, True).clamp(1) + 1e-8)	# [NK, NK]

        scores = list()
        lk_tilde = total_know_embedding	# [NK, DH]

        for k in range(k_hop):
            lk_tilde_1_rel = self.learn_kernel_matrix_rel[k](lk_tilde)			# [NK, KH]
            lk_tilde_2_rel = beta_rel_tilde.matmul(lk_tilde_1_rel)		# [NK, KH]
            lk_tilde_3_rel = lk_tilde_2_rel								# [NK, KH]
            lk_tilde_4_rel = self.learn_kernel_output_rel[k](lk_tilde_3_rel.relu())	# [NK, KH]

            lk_tilde_1_pre = self.learn_kernel_matrix_pre[k](lk_tilde)			# [NK, KH]
            lk_tilde_2_pre = beta_pre_tilde.matmul(lk_tilde_1_pre)		# [NK, KH]
            lk_tilde_3_pre = lk_tilde_2_pre								# [NK, KH]
            lk_tilde_4_pre = self.learn_kernel_output_pre[k](lk_tilde_3_pre.relu())	# [NK, KH]
            
            lk_tilde_1_sub = self.learn_kernel_matrix_sub[k](lk_tilde)			# [NK, KH]				
            lk_tilde_2_sub = beta_sub_tilde.matmul(lk_tilde_1_sub)		# [NK, KH]
            lk_tilde_3_sub = lk_tilde_2_sub								# [NK, KH]
            lk_tilde_4_sub = self.learn_kernel_output_sub[k](lk_tilde_3_sub.relu())	# [NK, KH]
        
            if k == 0:
                lk_tilde = lk_tilde_4_rel
            else:
                lk_tilde = lk_tilde + lk_tilde_4_rel

            lk_tilde = lk_tilde + lk_tilde_4_pre
            lk_tilde = lk_tilde + lk_tilde_4_sub

        learn_kernel_para = F.softplus(lk_tilde)*alpha	# [NK, KH]
        fk_tilde = total_know_embedding	# [NK, DH]

        for k in range(k_hop):
            fk_tilde_1_rel = self.forget_kernel_matrix_rel[k](fk_tilde)			# [NK, KH]
            fk_tilde_2_rel = beta_rel_tilde.matmul(fk_tilde_1_rel)		# [NK, KH]
            fk_tilde_3_rel = fk_tilde_2_rel								# [NK, KH]
            fk_tilde_4_rel = self.forget_kernel_output_rel[k](fk_tilde_3_rel.relu())	# [NK, KH]

            fk_tilde_1_pre = self.forget_kernel_matrix_pre[k](fk_tilde)			# [NK, KH]
            fk_tilde_2_pre = beta_pre_tilde.matmul(fk_tilde_1_pre)		# [NK, KH]
            fk_tilde_3_pre = fk_tilde_2_pre								# [NK, KH]
            fk_tilde_4_pre = self.forget_kernel_output_pre[k](fk_tilde_3_pre.relu())	# [NK, KH]
            
            fk_tilde_1_sub = self.forget_kernel_matrix_sub[k](fk_tilde)			# [NK, KH]				
            fk_tilde_2_sub = beta_sub_tilde.matmul(fk_tilde_1_sub)		# [NK, KH]
            fk_tilde_3_sub = fk_tilde_2_sub								# [NK, KH]
            fk_tilde_4_sub = self.forget_kernel_output_sub[k](fk_tilde_3_sub.relu())	# [NK, KH]
        
            if k == 0:
                fk_tilde = fk_tilde_4_rel
            else:
                fk_tilde = fk_tilde + fk_tilde_4_rel

            fk_tilde = fk_tilde + fk_tilde_4_pre
            fk_tilde = fk_tilde + fk_tilde_4_sub

        forget_kernel_para = F.softplus(fk_tilde)*alpha	# [NK, KH]
        learn_count = torch.zeros(B, NK).to(h.device).long()	# [B, NK]
        
        for i in range(S):
            h = h.clamp(min = -10, max = 10)
            # apply knowledge
            time = times[:, i]
            know = knows[:, i]						# [B, K]
            corr = corrs[:, i]						# [B]

            alpha = alpha_matrix[:, i]				# [B, NK]
            alpha_1 = alpha.unsqueeze(-1)			# [B, NK, 1]

            prob_emb = prob_embedding[:, i]			# [B, DH]
            know_emb = know_embedding[:, i].sum(-2)	# [B, DH]
            know_emb = know_emb/((know > 0).sum(-1, True).clamp(1) + 1e-8)	# [B, DH]
            know_prob_emb = torch.cat([know_emb, prob_emb], -1)	# [B, 2*DH]

            h_tilde = h						# [B, NK, KH]

            for k in range(k_hop):
                h_tilde_1_rel = self.agg_rel_matrix[k](h_tilde)					# [B, NK, KH]
                h_tilde_1_pre = self.agg_pre_matrix[k](h_tilde)					# [B, NK, KH]
                h_tilde_1_sub = self.agg_sub_matrix[k](h_tilde)					# [B, NK, KH]
                
                h_tilde_2_rel = h_tilde_1_rel*alpha_1							# [B, NK, KH]
                h_tilde_2_pre = h_tilde_1_pre*alpha_1							# [B, NK, KH]
                h_tilde_2_sub = h_tilde_1_sub*alpha_1							# [B, NK, KH]

                h_tilde_3_rel = beta_rel_tilde.matmul(h_tilde_2_rel)			# [B, NK, KH]
                h_tilde_3_pre = beta_pre_tilde.matmul(h_tilde_2_pre)			# [B, NK, KH]
                h_tilde_3_sub = beta_sub_tilde.matmul(h_tilde_2_sub)			# [B, NK, KH]
                
                h_tilde = h_tilde + h_tilde_3_rel
                h_tilde = h_tilde + h_tilde_3_pre
                h_tilde = h_tilde + h_tilde_3_sub								# [B, NK, KH]

            master = self.know_master_proj(h_tilde).squeeze(-1)			# [B, NK]
            master = master.gather(-1, know)							# [B, K]
            master = master.masked_fill(know == 0, 0)					# [B, K]
            master = master.sum(-1)										# [B]
            master = master / ((know > 0).sum(-1).clamp(1) + 1e-8)				# [B]
            diff = self.prob_diff_mlp(know_prob_emb).squeeze(-1)		# [B]

            score = (master - diff).sigmoid()							# [B]
            scores.append(score)

            # knowledge gain and loss
            # know: [B, K]

            know_index = know[:, :, None].expand(B, K, KH)			# [B, K, KH]
            target_h = h.gather(-2, know_index)						# [B, K, KH]
            know_prob_emb_1 = know_prob_emb.unsqueeze(-2).expand(B, K, 2*DH)	# [B, K, 2*DH]
            
            gain = self.gain_ffn(torch.cat([know_prob_emb_1, target_h], -1))	# [B, K, KH]
            gain_1 = torch.zeros_like(h)								# [B, NK, KH]	
            total_gain = gain_1.scatter(-2, know_index, gain)				# [B, NK, KH]

            for k in range(k_hop):
                total_gain_1_rel = self.gain_matrix_rel[k](total_gain)			# [B, NK, KH]
                total_gain_2_rel = beta_rel_tilde.matmul(total_gain_1_rel)		# [B, NK, KH]
                total_gain_3_rel = total_gain_2_rel*alpha_1		# [B, NK, KH]
                total_gain_4_rel = self.gain_output_rel[k](total_gain_3_rel.relu())	# [B, NK, KH]

                total_gain_1_pre = self.gain_matrix_pre[k](total_gain)			# [B, NK, KH]
                total_gain_2_pre = beta_pre_tilde.matmul(total_gain_1_pre)		# [B, NK, KH]
                total_gain_3_pre = total_gain_2_pre*alpha_1		# [B, NK, KH]
                total_gain_4_pre = self.gain_output_pre[k](total_gain_3_pre.relu())	# [B, NK, KH]
                
                total_gain_1_sub = self.gain_matrix_sub[k](total_gain)			# [B, NK, KH]				
                total_gain_2_sub = beta_sub_tilde.matmul(total_gain_1_sub)		# [B, NK, KH]
                total_gain_3_sub = total_gain_2_sub*alpha_1		# [B, NK, KH]
                total_gain_4_sub = self.gain_output_sub[k](total_gain_3_sub.relu())	# [B, NK, KH]
            
                total_gain = total_gain + total_gain_4_rel
                total_gain = total_gain + total_gain_4_pre
                total_gain = total_gain + total_gain_4_sub

            total_gain = total_gain.relu()

            loss = self.loss_ffn(torch.cat([know_prob_emb_1, target_h], -1))	# [B, K, KH]
            loss_1 = torch.zeros_like(h)								# [B, NK, KH]
            total_loss = loss_1.scatter(-2, know_index, loss)

            for k in range(k_hop):
                total_loss_1_rel = self.loss_matrix_rel[k](total_loss)			# [B, NK, KH]
                total_loss_2_rel = beta_rel_tilde.matmul(total_loss_1_rel)		# [B, NK, KH]
                total_loss_3_rel = total_loss_2_rel*alpha_1		# [B, NK, KH]
                total_loss_4_rel = self.loss_output_rel[k](total_loss_3_rel.relu())	# [B, NK, KH]

                total_loss_1_pre = self.loss_matrix_pre[k](total_loss)			# [B, NK, KH]
                total_loss_2_pre = beta_pre_tilde.matmul(total_loss_1_pre)		# [B, NK, KH]
                total_loss_3_pre = total_loss_2_pre*alpha_1		# [B, NK, KH]
                total_loss_4_pre = self.loss_output_pre[k](total_loss_3_pre.relu())	# [B, NK, KH]
                
                total_loss_1_sub = self.loss_matrix_sub[k](total_loss)			# [B, NK, KH]				
                total_loss_2_sub = beta_sub_tilde.matmul(total_loss_1_sub)		# [B, NK, KH]
                total_loss_3_sub = total_loss_2_sub*alpha_1		# [B, NK, KH]
                total_loss_4_sub = self.loss_output_sub[k](total_loss_3_sub.relu())	# [B, NK, KH]
            
                total_loss = total_loss + total_loss_4_rel
                total_loss = total_loss + total_loss_4_pre
                total_loss = total_loss + total_loss_4_sub

            total_loss = total_loss.relu()

            corr_1 = corr[:, None, None]
            h = h + corr_1*total_gain - (~corr_1*total_loss)	# [B, NK, KH]
            learn_count = learn_count + ((corr_1*total_gain) > 0).any(-1).long()	# [B, NK]

            if i != S - 1:
                new_know = knows[:, i + 1]					# [B, K]
                new_time = times[:, i + 1]					# [B, K]

                new_know_index = new_know[:, :, None].expand(B, K, KH)			# [B, K, KH]
                new_target_h = h.gather(-2, new_know_index)						# [B, K, KH]
                total_target_h = torch.cat([target_h, new_target_h], -2)		# [B, 2K, KH]
                total_know_index = torch.cat([know_index, new_know_index], -2)	# [B, 2K, KH]

                new_prob_emb = prob_embedding[:, i + 1]		# [B, DH]
                new_know_emb = know_embedding[:, i + 1].sum(-2)	# [B, DH]
                new_know_emb = new_know_emb/((new_know > 0).sum(-1, True).clamp(1) + 1e-8)	# [B, DH]

                new_know_prob_emb = torch.cat([new_know_emb, new_prob_emb], -1)	# [B, 2*DH]
                total_know_prob_emb = torch.cat([know_prob_emb, new_know_prob_emb], -1)	# [B, 4*DH]
                total_know_prob_emb_1 = total_know_prob_emb.unsqueeze(-2).expand(B, 2*K, 4*DH)	# [B, 2*K, 4*DH]

                learn_input = torch.cat([total_know_prob_emb_1, total_target_h], -1)	# [B, 2*K, 4*DH+KH]
                decision = self.decision_mlp(learn_input) # [B, 2*K, 2]
                decision_gumbel_mask = F.gumbel_softmax(decision, tau = tau, hard = True, dim = -1)	# [B, 2*K, 2]
                decision_gumbel_mask_1 = decision_gumbel_mask[:, :, :1]	# [B, 2*K, 1]
                decision_gumbel_mask_2 = decision_gumbel_mask[:, :, 1:]	# [B, 2*K, 1]
                learn = self.learn_mlp(learn_input)							# [B, 2*K, KH]
                learn_1 = torch.zeros_like(learn)							# [B, 2*K, KH]
                learn_2 = torch.zeros_like(h)								# [B, NK, KH]
                learn_3 = learn*decision_gumbel_mask_1 + learn_1*decision_gumbel_mask_2	# [B, 2*K, KH]

                total_learn = learn_2.scatter(-2, total_know_index, learn_3)	# [B, NK, KH]

                for k in range(k_hop):
                    total_learn_1_rel = self.learn_matrix_rel[k](total_learn)			# [B, NK, KH]
                    total_learn_2_rel = beta_rel_tilde.matmul(total_learn_1_rel)		# [B, NK, KH]
                    total_learn_3_rel = total_learn_2_rel
                    total_learn_4_rel = self.learn_output_rel[k](total_learn_3_rel.relu())	# [B, NK, KH]

                    total_learn_1_pre = self.learn_matrix_pre[k](total_learn)			# [B, NK, KH]
                    total_learn_2_pre = beta_pre_tilde.matmul(total_learn_1_pre)		# [B, NK, KH]
                    total_learn_3_pre = total_learn_2_pre
                    total_learn_4_pre = self.learn_output_pre[k](total_learn_3_pre.relu())	# [B, NK, KH]
                    
                    total_learn_1_sub = self.learn_matrix_sub[k](total_learn)			# [B, NK, KH]				
                    total_learn_2_sub = beta_sub_tilde.matmul(total_learn_1_sub)		# [B, NK, KH]
                    total_learn_3_sub = total_learn_2_sub
                    total_learn_4_sub = self.learn_output_sub[k](total_learn_3_sub.relu())	# [B, NK, KH]
                
                    total_learn = total_learn + total_learn_4_rel
                    total_learn = total_learn + total_learn_4_pre
                    total_learn = total_learn + total_learn_4_sub
                
                total_learn = total_learn.relu()	# [B, NK, KH]
                history_gain = (h - h_initial).clamp(0)	# [B, NK, KH]

                learn_kernel_para_1 = learn_kernel_para.unsqueeze(0).expand(B, NK, KH)	# [B, NK, KH]
                forget_kernel_para_1 = forget_kernel_para.unsqueeze(0).expand(B, NK, KH)# [B, NK, KH]

                delta_time = (new_time - time).clamp(0).float()[:, None, None] # [B, 1, 1]

                learn_exp = (-(learn_count[:, :, None].float() + 1)*delta_time*learn_kernel_para_1).exp()	# [B, NK, KH]
                forget_exp = (-(learn_count[:, :, None].float() + 1)*delta_time*forget_kernel_para_1).exp()	# [B, NK, KH]

                h = h + (1 - learn_exp)*total_learn	# [B, NK, KH]
                gain_after_forget = history_gain*forget_exp*(total_learn == 0).all(-1, True)	# [B, NK, KH]
                h = h - (history_gain - gain_after_forget)*(total_learn == 0).all(-1, True)	# [B, NK, KH]
            
                learn_count = learn_count + ((total_learn) > 0).any(-1).long()	# [B, NK]
        # [B, S]
        scores = torch.stack(scores, -1)							
        return scores
        
    def get_predict_score(self, batch, seq_start=2):
        mask_seq = torch.ne(batch["mask_seq"], 0)
        # predict_score_batch的shape必须为(bs, seq_len-1)，其中第二维的第一个元素为对序列第二题的预测分数
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
