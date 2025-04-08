import torch


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):  
    # Equation (4)
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True) 
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True) 
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2) 

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret 