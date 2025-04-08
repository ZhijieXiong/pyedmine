import torch
import torch.nn as nn
import torch.nn.functional as F

from edmine.model.module.calculation import wasserstein_distance_matmul


def binary_cross_entropy(predict_score, ground_truth, device):
    if device == "mps":
        return F.binary_cross_entropy(predict_score.float(), ground_truth.float())
    else:
        return F.binary_cross_entropy(predict_score.double(), ground_truth.double())
    
    
def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24))
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)
    return ret


def d2s_1overx(distance):
    return 1/(1+distance)


class WassersteinNCELoss(nn.Module):
    """UKT"""
    def __init__(self, temperature):
        super(WassersteinNCELoss, self).__init__()
        self.temperature = temperature
        self.activation = nn.ELU()

    def forward(self, batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov):
        batch_sample_one_cov = self.activation(batch_sample_one_cov) + 1
        batch_sample_two_cov = self.activation(batch_sample_two_cov) + 1
        sim11 = d2s_1overx(wasserstein_distance_matmul(batch_sample_one_mean, batch_sample_one_cov, batch_sample_one_mean, batch_sample_one_cov)) / self.temperature
        sim22 = d2s_1overx(wasserstein_distance_matmul(batch_sample_two_mean, batch_sample_two_cov, batch_sample_two_mean, batch_sample_two_cov)) / self.temperature
        sim12 = -d2s_1overx(wasserstein_distance_matmul(batch_sample_one_mean, batch_sample_one_cov, batch_sample_two_mean, batch_sample_two_cov)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = F.cross_entropy(logits, labels)
        return nce_loss
