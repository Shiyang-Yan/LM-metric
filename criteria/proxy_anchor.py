import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

import torch, torch.nn as nn



"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = opt.n_classes
        self.sz_embed = opt.embed_dim
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        self.proxies_rnn = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = opt.n_classes
        self.sz_embed = opt.embed_dim

        self.mrg = 0.1
        self.alpha = 32
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def forward(self, batch, rnn_outputs, labels, phi, **kwargs):
        P = self.proxies
        X= batch
        T =labels
        cos = F.linear((X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = phi[0].unsqueeze(1)*torch.exp((-self.alpha *(cos - (self.mrg))))
        neg_exp = phi[0].unsqueeze(1)*torch.exp((self.alpha * (cos + (self.mrg))))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
##################################
        P = self.proxies_rnn
        X = rnn_outputs
        T = labels
        cos = F.linear((X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = phi[0].unsqueeze(1)*torch.exp(-self.alpha * (cos - (self.mrg)))
        neg_exp = phi[0].unsqueeze(1)*torch.exp(self.alpha * (cos + (self.mrg)))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss_rnn = pos_term + neg_term

        return loss + loss_rnn