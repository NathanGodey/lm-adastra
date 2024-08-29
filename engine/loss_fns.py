import torch

def cwt_loss(inputs, positive, negative, temperature=1.):
    self_dist = torch.exp((inputs*positive).sum(-1)/temperature)
    neg_dist = torch.exp(torch.mm(inputs, negative.T)/temperature).sum(-1)
    return - (self_dist/(self_dist + neg_dist + 1e-9)).log().mean()

# def cwt_loss(inputs, positive, negative, weights, temperature=1.):
#     self_dist = torch.exp((inputs*positive).sum(-1)/temperature)
#     neg_dist = (weights * torch.exp(torch.mm(inputs, negative.T)/temperature)).sum(-1)
#     return - (self_dist/(self_dist + neg_dist + 1e-9)).log().mean()