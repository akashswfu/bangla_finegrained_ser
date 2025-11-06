import torch, math, torch.nn.functional as F
def log_mean_exp_over_time(Z, mask=None):
    if mask is not None:
        Zm = Z.masked_fill(~mask.unsqueeze(-1), torch.finfo(Z.dtype).min)
        lse = torch.logsumexp(Zm, 1)
        denom = mask.sum(1, keepdim=True).clamp_min(1)
        return lse - torch.log(denom)
    lse = torch.logsumexp(Z,1); return lse - math.log(Z.size(1))
def bag_loss(Z, y, mask=None):
    return F.cross_entropy(log_mean_exp_over_time(Z, mask), y)
def temporal_consistency_loss(Z, mask=None):
    P=F.softmax(Z,-1); diff=(P[:,1:,:]-P[:,:-1,:]).abs()
    if mask is not None:
        m=mask[:,1:] & mask[:,:-1]; diff=diff*m.unsqueeze(-1).float(); denom=m.sum().clamp_min(1)
    else: denom=diff.numel()/diff.shape[-1]
    return diff.sum()/denom
def change_sparsity_loss(Z, mask=None):
    P=F.softmax(Z,-1); a=P.argmax(-1); ch=(a[:,1:]!=a[:,:-1]).float()
    if mask is not None:
        m=(mask[:,1:] & mask[:,:-1]).float(); ch=ch*m; denom=m.sum().clamp_min(1)
    else: denom=ch.numel()
    return ch.sum()/denom
