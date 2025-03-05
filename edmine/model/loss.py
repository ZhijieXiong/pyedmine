from torch.nn.functional import binary_cross_entropy as binary_cross_entropy_


def binary_cross_entropy(predict_score, ground_truth, device):
    if device == "mps":
        return binary_cross_entropy_(predict_score.float(), ground_truth.float())
    else:
        return binary_cross_entropy_(predict_score.double(), ground_truth.double())
