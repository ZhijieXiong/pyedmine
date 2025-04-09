import torch
from torch import nn

from edmine.model.sequential_kt_model.DLSequentialKTModel import DLSequentialKTModel
from edmine.model.module.EmbedLayer import EmbedLayer, CosinePositionalEmbedding
from edmine.model.module.Transformer import TransformerLayer4UKT
from edmine.model.module.PredictorLayer import PredictorLayer
from edmine.model.loss import WassersteinNCELoss, binary_cross_entropy


class UKT(nn.Module, DLSequentialKTModel):
    model_name = "UKT"
    
    def __init__(self, params, objects):
        super().__init__()
        self.params = params
        self.objects = objects

        model_config = self.params["models_config"]["UKT"]
        temperature = model_config["temperature"]
        self.wloss = WassersteinNCELoss(temperature)
        self.embed_layer = EmbedLayer(model_config["embed_config"])
        self.model = Architecture(params)
        self.out = PredictorLayer(model_config["predictor_config"])

    def base_emb(self, batch, aug=False):
        separate_qa = self.params["models_config"]["UKT"]["separate_qa"]
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]
        num_concept = self.objects["dataset"]["q_table"].shape[0]
        
        if aug:
            correctness_seq = batch["aug_correctness_seq"]
        else:
            correctness_seq = batch["correctness_seq"]
        
        concept_mean_emb = self.embed_layer.get_emb_fused1(
            "concept_mean", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        concept_cov_emb = self.embed_layer.get_emb_fused1(
            "concept_cov", q2c_transfer_table, q2c_mask_table, batch["question_seq"])

        if separate_qa:
            interaction_seq = num_concept * correctness_seq.unsqueeze(-1)
            interaction_mean_emb = self.embed_layer.get_emb_fused1(
                "interaction_mean", q2c_transfer_table, q2c_mask_table, batch["question_seq"], other_item_index=interaction_seq)
            interaction_cov_emb  = self.embed_layer.get_emb_fused1(
                "interaction_cov", q2c_transfer_table, q2c_mask_table, batch["question_seq"], other_item_index=interaction_seq)
        else:
            interaction_mean_emb = self.embed_layer.get_emb("interaction_mean", correctness_seq) + concept_mean_emb
            interaction_cov_emb  = self.embed_layer.get_emb("interaction_cov", correctness_seq)  + concept_cov_emb

        return concept_mean_emb, concept_cov_emb, interaction_mean_emb, interaction_cov_emb
    
    def get_predict_score(self, batch, seq_start=2):        
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        # Generate stochastic embeddings for questions and responses # Equation (1)
        c_mean_emb,c_cov_emb, ca_mean_emb, ca_cov_emb = self.base_emb(batch)
        aug_c_mean_emb,aug_c_cov_emb, aug_ca_mean_emb, aug_ca_cov_emb = self.base_emb(batch, aug=True)

        c_var_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        q_diff_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        c_mean_emb = c_mean_emb + q_diff_emb * c_var_emb 
        c_cov_emb = c_cov_emb + q_diff_emb * c_var_emb
        ca_var_emb = self.embed_layer.get_emb("interaction_var", batch["correctness_seq"])
        ca_mean_emb = ca_mean_emb + q_diff_emb * (ca_var_emb + c_var_emb)
        ca_cov_emb = ca_cov_emb + q_diff_emb * (ca_var_emb + c_var_emb)

        aug_ca_var_emb = self.embed_layer.get_emb("interaction_var", batch["aug_correctness_seq"])
        aug_c_mean_emb = aug_c_mean_emb + q_diff_emb * c_var_emb
        aug_c_cov_emb = aug_c_cov_emb + q_diff_emb * c_var_emb
        aug_ca_mean_emb = aug_ca_mean_emb + q_diff_emb * (aug_ca_var_emb+c_var_emb)
        aug_ca_cov_emb = aug_ca_cov_emb + q_diff_emb * (aug_ca_var_emb+c_var_emb)

        mean_d_output,cov_d_output = self.model(c_mean_emb,c_cov_emb, ca_mean_emb,ca_cov_emb)
     
        concat_q = torch.cat([mean_d_output,cov_d_output,c_mean_emb,c_cov_emb], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        predict_score_batch = m(output)[:, 1:]
        
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        return {
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_predict_loss(self, batch, seq_start=2):        
        q2c_transfer_table = self.objects["dataset"]["q2c_transfer_table"]
        q2c_mask_table = self.objects["dataset"]["q2c_mask_table"]

        # Generate stochastic embeddings for questions and responses # Equation (1)
        c_mean_emb,c_cov_emb, ca_mean_emb, ca_cov_emb = self.base_emb(batch)
        aug_c_mean_emb,aug_c_cov_emb, aug_ca_mean_emb, aug_ca_cov_emb = self.base_emb(batch, aug=True)

        c_var_emb = self.embed_layer.get_emb_fused1("concept_var", q2c_transfer_table, q2c_mask_table, batch["question_seq"])
        q_diff_emb = self.embed_layer.get_emb("question_diff", batch["question_seq"])
        c_mean_emb = c_mean_emb + q_diff_emb * c_var_emb 
        c_cov_emb = c_cov_emb + q_diff_emb * c_var_emb
        ca_var_emb = self.embed_layer.get_emb("interaction_var", batch["correctness_seq"])
        ca_mean_emb = ca_mean_emb + q_diff_emb * (ca_var_emb + c_var_emb)
        ca_cov_emb = ca_cov_emb + q_diff_emb * (ca_var_emb + c_var_emb)
        mean_d_output,cov_d_output = self.model(c_mean_emb,c_cov_emb, ca_mean_emb,ca_cov_emb)

        aug_ca_var_emb = self.embed_layer.get_emb("interaction_var", batch["aug_correctness_seq"])
        aug_c_mean_emb = aug_c_mean_emb + q_diff_emb * c_var_emb
        aug_c_cov_emb = aug_c_cov_emb + q_diff_emb * c_var_emb
        aug_ca_mean_emb = aug_ca_mean_emb + q_diff_emb * (aug_ca_var_emb+c_var_emb)
        aug_ca_cov_emb = aug_ca_cov_emb + q_diff_emb * (aug_ca_var_emb+c_var_emb)
        mean_d2_output, cov_d2_output = self.model(aug_c_mean_emb,aug_c_cov_emb, aug_ca_mean_emb,aug_ca_cov_emb)
        
        mask = batch["mask_seq"].unsqueeze(-1)
        pooled_mean_d_output = torch.mean(mean_d_output * mask,dim = 1)
        pooled_cov_d_output = torch.mean(cov_d_output * mask,dim = 1)
        pooled_mean_d2_output = torch.mean(mean_d2_output * mask,dim = 1)
        pooled_cov_d2_output = torch.mean(cov_d2_output * mask,dim = 1)
        cl_loss = self.wloss(pooled_mean_d_output, pooled_cov_d_output, pooled_mean_d2_output, pooled_cov_d2_output)
                
        concat_q = torch.cat([mean_d_output,cov_d_output,c_mean_emb,c_cov_emb], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        predict_score_batch = m(output)[:, 1:]
        mask_seq = torch.ne(batch["mask_seq"], 0)
        predict_score = torch.masked_select(predict_score_batch[:, seq_start-2:], mask_seq[:, seq_start-1:])
        ground_truth = torch.masked_select(batch["correctness_seq"][:, seq_start-1:], mask_seq[:, seq_start-1:])
        predict_loss = binary_cross_entropy(predict_score, ground_truth, self.params["device"])
        
        loss = predict_loss + self.params["loss_config"]["cl loss"] * cl_loss
        num_sample = torch.sum(batch["mask_seq"][:, seq_start-1:]).item()
        batch_size = mask_seq.shape[0]
        return {
            "total_loss": loss,
            "losses_value": {
                "predict loss": {
                    "value": predict_loss.detach().cpu().item() * num_sample,
                    "num_sample": num_sample
                },
                "cl loss": {
                    "value": cl_loss.detach().cpu().item() * batch_size,
                    "num_sample": batch_size
                }
            },
            "predict_score": predict_score,
            "predict_score_batch": predict_score_batch
        }

    def get_knowledge_state(self, batch):
        pass
            

class Architecture(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.parmas = params
        
        model_config = params["models_config"]["UKT"]
        dim_model = model_config["dim_model"]
        seq_len = model_config["seq_len"]
        num_block = model_config["num_block"]
        
        self.position_mean_embeddings = CosinePositionalEmbedding(dim_model, max_seq_len=seq_len)
        self.position_cov_embeddings = CosinePositionalEmbedding(dim_model, max_seq_len=seq_len)
        self.blocks_2 = nn.ModuleList([TransformerLayer4UKT(params) for _ in range(num_block)])

    def forward(self, q_mean_embed_data,q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data):
        # Equation (2)
        mean_q_posemb = self.position_mean_embeddings(q_mean_embed_data)
        cov_q_posemb = self.position_cov_embeddings(q_cov_embed_data)

        q_mean_embed_data = q_mean_embed_data + mean_q_posemb
        q_cov_embed_data = q_cov_embed_data + cov_q_posemb

        qa_mean_posemb = self.position_mean_embeddings(qa_mean_embed_data)
        qa_cov_posemb = self.position_cov_embeddings(qa_cov_embed_data)

        qa_mean_embed_data = qa_mean_embed_data + qa_mean_posemb
        qa_cov_embed_data = qa_cov_embed_data + qa_cov_posemb
        
        # Equation (3)
        elu_act = torch.nn.ELU()
        q_mean_embed_data = q_mean_embed_data
        q_cov_embed_data = elu_act(q_cov_embed_data) + 1
        qa_mean_embed_data = qa_mean_embed_data
        qa_cov_embed_data = elu_act(qa_cov_embed_data) + 1

        mean_qa_pos_embed = qa_mean_embed_data
        cov_qa_pos_embed = qa_cov_embed_data

        mean_q_pos_embed = q_mean_embed_data
        cov_q_pos_embed = q_cov_embed_data

        # y = qa_pos_embed
        y_mean = mean_qa_pos_embed
        y_cov  = cov_qa_pos_embed

        # x = q_pos_embed
        x_mean = mean_q_pos_embed
        x_cov = cov_q_pos_embed

        # encoder
        for block in self.blocks_2:
            x_mean,x_cov = block(mask=0, query_mean=x_mean, query_cov = x_cov, key_mean=x_mean,key_cov=x_cov, values_mean=y_mean, values_cov = y_cov, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever

        return x_mean,x_cov

