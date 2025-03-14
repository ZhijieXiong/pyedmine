import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from edmine.model.exercise_recommendation_model.DLExerciseRecommendationModel import DLExerciseRecommendationModel


def TransE(head, relation, tail, gamma):
    return gamma - torch.norm((head + relation) - tail, p=2)


class KG4EX(nn.Module, DLExerciseRecommendationModel):
    model_name = "KG4EX"

    def __init__(self, params, objects):
        super(KG4EX, self).__init__()
        self.params = params
        self.objects = objects

        model_config = params["models_config"]["KG4EX"]
        model_selection = model_config["model_selection"]
        num_entity = len(self.objects["dataset"]["entity2id"])
        num_relation = len(self.objects["dataset"]["relation2id"])
        dim = model_config["dim"]
        gamma = model_config["gamma"]
        double_entity_embedding = model_config["double_entity_embedding"]
        double_relation_embedding = model_config["double_relation_embedding"]
        epsilon = model_config["epsilon"]

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + epsilon) / dim]),
            requires_grad=False
        )
        dim_entity = dim * 2 if double_entity_embedding else dim
        dim_relation = dim * 2 if double_relation_embedding else dim
        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, dim_entity))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, dim_relation))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_selection not in ['TransE', 'RotatE']:
            raise ValueError(f'model {model_selection} not supported')

        if model_selection == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

    def forward(self, sample, mode='single'):
        if mode == 'single':
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_selection = self.params["models_config"]["KG4EX"]["model_selection"]
        if model_selection == "TransE":
            score = self.TransE(head, relation, tail)
        elif model_selection == "RotatE":
            score = self.RotatE(head, relation, tail, mode)
        else:
            raise ValueError(f'{model_selection} not supported')

        return score

    def TransE(self, head, relation, tail):
        return self.gamma.item() - torch.norm(head + relation - tail, p=1, dim=2)

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def train_one_step(self, one_step_data):
        model_config = self.params["models_config"]["KG4EX"]
        negative_adversarial_sampling = model_config["negative_adversarial_sampling"]
        uni_weight = model_config["uni_weight"]
        adversarial_temperature = model_config["adversarial_temperature"]
        w_reg_loss = self.params["loss_config"]["regularization loss"]

        self.train()
        positive_sample, negative_sample, subsampling_weight, mode = one_step_data
        negative_score = self.forward((positive_sample, negative_sample), mode=mode[0])

        if negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = self.forward(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        losses_value = {
            "positive sample loss": {
                "value": positive_sample_loss.detach().cpu().item(),
                "num_sample": 1
            },
            "negative sample loss": {
                "value": negative_sample_loss.detach().cpu().item(),
                "num_sample": 1
            },
        }
        if w_reg_loss > 0:
            # Use L3 regularization for ComplEx and DistMult
            re_loss = w_reg_loss * (
                    self.entity_embedding.norm(p=3) ** 3 +
                    self.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + re_loss
            losses_value["regularization loss"] = {
                "value": negative_sample_loss.detach().cpu().item(),
                "num_sample": 1
            }
        return {
            "total_loss": loss,
            "losses_value": losses_value
        }

    def data2batches(self, data, batch_size, show_process_bar=False):
        batches = []
        batch = []
        for user_id, user_data in data.items():
            user_data["user_id"] = user_id
            if len(batch) < batch_size:
                batch.append(user_data)
            else:
                batches.append(batch)
                batch = [user_data]
        if len(batch) > 0:
            batches.append(batch)
        batches_tensor = []
        for i, batch in enumerate(batches):
            batches_tensor.append({
                "user_id": torch.tensor([x["user_id"] for x in batch]).long().to(self.params["device"]),
                "mlkc": torch.tensor([x["mlkc"] for x in batch]).long().to(self.params["device"]),
                "pkc": torch.tensor([x["pkc"] for x in batch]).long().to(self.params["device"]),
                "efr": torch.tensor([x["efr"] for x in batch]).long().to(self.params["device"])
            })
        if show_process_bar:
            return tqdm(batches_tensor, desc="inferencing: ")
        else:
            return batches_tensor

    def get_top_ns(self, data, top_ns, batch_size, show_process_bar=False):
        entity2id = self.objects["dataset"]["entity2id"]
        q_table = self.objects["dataset"]["q_table"]
        num_question, num_concept = q_table.shape[0], q_table.shape[1]
        gamma = self.params["models_config"]["KG4EX"]["gamma"]
        max_top_n = max(top_ns)

        _, user_mlkc, (user_pkc, user_efr) = data
        users_data = {}
        for user_id in user_mlkc.keys():
            users_data[user_id] = {
                "mlkc": list(map(lambda x: int(x * 100), user_mlkc[user_id])),
                "pkc": list(map(lambda x: 101 + int(x * 100), user_pkc[user_id])),
                "efr": list(map(lambda x: 202 + int(x * 100), user_efr[user_id])),
            }

        # 矩阵运算推理
        all_kc_id = []
        for i in range(num_concept):
            c_id = entity2id[f"kc{i}"]
            all_kc_id.append(c_id)
        all_kc_id = torch.tensor(all_kc_id).long().to(self.params["device"])
        all_que_id = []
        for j in range(num_question):
            q_id = entity2id[f"ex{j}"]
            all_que_id.append(q_id)
        all_que_id = torch.tensor(all_que_id).long().to(self.params["device"])
        batches = self.data2batches(users_data, batch_size, show_process_bar)
        users_rec_questions = {}
        for batch in batches:
            batch_size = batch["mlkc"].shape[0]
            num_batch_q = 1000
            scores = []
            for i in range(0, num_question, num_batch_q):
                batch_que_id = all_que_id[i:i+num_batch_q]
                q_batch_size = batch_que_id.shape[0]
                
                rec_emb4fr1 = self.relation_embedding[303].expand(batch_size, q_batch_size, num_concept, -1)
                rec_emb4fr2 = self.relation_embedding[303].expand(batch_size, q_batch_size, -1)

                kc_emb = self.entity_embedding[all_kc_id].expand(batch_size, -1, -1)
                que_emb = self.entity_embedding[batch_que_id].expand(batch_size, -1, -1)
                mlkc_emb = self.relation_embedding[batch["mlkc"]]
                pkc_emb = self.relation_embedding[batch["pkc"]]
                efr_emb = self.relation_embedding[batch["efr"][:, i:i+q_batch_size]]

                mlkc_emb4que = (kc_emb + mlkc_emb).repeat_interleave(q_batch_size, dim=0).view(batch_size, q_batch_size, num_concept, -1)
                pkc_emb4que = (kc_emb + pkc_emb).repeat_interleave(q_batch_size, dim=0).view(batch_size, q_batch_size, num_concept, -1)
                que_emb_extend = que_emb.repeat_interleave(num_concept, dim=1).view(batch_size, q_batch_size, num_concept, -1)

                # 下面就是TransE的操作
                fr1_emb1 = mlkc_emb4que + rec_emb4fr1 - que_emb_extend
                fr1_sum1 = gamma - fr1_emb1.norm(dim=-1)
                fr1_mlkc = fr1_sum1.sum(dim=-1)
                # 同理
                fr1_emb2 = pkc_emb4que + rec_emb4fr1 - que_emb_extend
                fr1_sum2 = gamma - fr1_emb2.norm(dim=-1)
                fr1_pkc = fr1_sum2.sum(dim=-1)
                fr1 = fr1_mlkc + fr1_pkc
                # 原代码中TransE(ej_embedding + efr_embedding, rec_embedding, e)的操作中ej_embedding和e是相同的emb
                fr2_emb = efr_emb + rec_emb4fr2
                fr2 = gamma - fr2_emb.norm(dim=-1)
                
                # shape: (batch_size, num_batch_q)
                scores.append(((fr1 / num_concept) + fr2).detach().cpu())
            scores = torch.cat(scores, dim=1)
            _, indices = torch.topk(scores, max_top_n, dim=1)
            batch_user_ids = batch["user_id"].detach().cpu().numpy()
            for user_id, rec_q_ids in zip(batch_user_ids, indices.detach().cpu().numpy()):
                rec_questions = {top_n: [] for top_n in top_ns}
                for i, rec_q_id in enumerate(rec_q_ids):
                    if i >= max_top_n:
                        break
                    for top_n in top_ns:
                        if i < top_n:
                            rec_questions[top_n].append(rec_q_id)
                users_rec_questions[user_id] = rec_questions

        # 循环的方式推理
        # rec_embedding = self.relation_embedding[303]
        # users_rec_questions = {}
        # for user_id, user_data in users_data.items():
        #     rec_questions = {top_n: [] for top_n in top_ns}
        #     scores = []
        #     s_mlkc_list = []
        #     s_pkc_list = []

        #     for i in range(num_concept):
        #         c_id = entity2id[f"kc{i}"]
        #         mlkc_id = user_data["mlkc"][i]
        #         pkc_id = user_data["pkc"][i]

        #         kc_embedding = self.entity_embedding[c_id]
        #         mlkc_embedding = self.relation_embedding[mlkc_id]
        #         pkc_embedding = self.relation_embedding[pkc_id]

        #         s_mlkc_list.append(kc_embedding + mlkc_embedding)
        #         s_pkc_list.append(kc_embedding + pkc_embedding)

        #     for j in range(num_question):
        #         q_id = entity2id[f"ex{j}"]
        #         efr_id = user_data["efr"][j]

        #         e = self.entity_embedding[q_id]
        #         fr1 = 0.0

        #         for s_mlkc in s_mlkc_list:
        #             fr1 += TransE(s_mlkc, rec_embedding, e, gamma)

        #         for s_pkc in s_pkc_list:
        #             fr1 += TransE(s_pkc, rec_embedding, e, gamma)

        #         ej_embedding = self.entity_embedding[q_id]
        #         efr_embedding = self.relation_embedding[efr_id]
        #         s_efr = ej_embedding + efr_embedding
        #         fr2 = TransE(s_efr, rec_embedding, e, gamma)

        #         scores.append((j, (fr1 / num_concept + fr2).detach().cpu().item()))

        #     question_sorted = list(map(lambda x: x[0], sorted(scores, key=lambda x: x[1], reverse=True)))
        #     for i, rec_q_id in enumerate(question_sorted):
        #         if i >= max_top_n:
        #             break
        #         for top_n in top_ns:
        #             if i < top_n:
        #                 rec_questions[top_n].append(rec_q_id)
        #     users_rec_questions[user_id] = rec_questions

        return users_rec_questions
