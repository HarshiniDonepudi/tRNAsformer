import torch

def RankLoss(features, mask, inv_mask, batch_size):
    features_norm = torch.div(features, torch.norm(features, dim=1, keepdim=True))
    features_sim = torch.matmul(features_norm, features_norm.T)
    features_exp = torch.exp(2. * features_sim)
    denominator = torch.sum(features_exp * inv_mask.cuda(), dim=1, keepdim=True)
    denominator = denominator.repeat((1, batch_size))
    denominator = denominator + features_exp
    ratio = torch.div(features_exp, denominator)
    loss = torch.mean(-torch.log(ratio) * mask.cuda())
    return loss