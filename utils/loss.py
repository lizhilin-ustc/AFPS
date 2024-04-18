import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss

# class CrossEntropyLoss1(nn.Module):
#     def __init__(self):
#         super(CrossEntropyLoss1, self).__init__()
#         self.ce_criterion = nn.BCELoss()

#     def forward(self, logits, label):
#         label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
#         loss = -torch.mean(torch.sum(label * F.log(logits, dim=1), dim=1), dim=0)
#         return loss

class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7

        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)

        return first_term + second_term

class GeneralizedCE_Mask(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE_Mask, self).__init__()

    def forward(self, logits, label, mask):

        # print(logits.shape)
        # print(label.shape)
        # exit()
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label * mask, dim=1) + 1e-7
        neg_factor = torch.sum((1 - label) * mask, dim=1) + 1e-7

        # print(pos_factor, 2222)

        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label * mask, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label) * mask, dim=1)/neg_factor)

        return first_term + second_term

class BCE(nn.Module):
    def __init__(self, ):
        super(BCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7

        first_term = - torch.mean(torch.sum((logits + 1e-7).log() * label, dim=1) / pos_factor)
        second_term = - torch.mean(torch.sum((1 - logits + 1e-7).log() * (1 - label), dim=1) / neg_factor)

        return first_term + second_term


class BCE1(nn.Module):
    def __init__(self, ):
        super(BCE1, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        factor = torch.sum(label, dim=1) + 1e-7
        loss = - torch.mean(torch.sum((logits + 1e-7).log() * label, dim=1) / factor)

        return loss

class Focal(nn.Module):
    def __init__(self, gamma):
        super(Focal, self).__init__()
        self.gamma = gamma

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]

        factor = torch.sum(label, dim=1) + 1e-7
        loss = - torch.mean(torch.sum((logits + 1e-7).log() * label * ((1 - logits)**self.gamma), dim=1) / factor)

        return loss


class GeneralizedCE2(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE2, self).__init__()


    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7

        pos_numerators = 1. - torch.pow(torch.sum(label * logits, dim=1), self.q)
        pos_denominators = 1 - logits.pow(self.q).sum(dim=1)
        pos_ngce = pos_numerators / pos_denominators

        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)

        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)

        return first_term + second_term


class GeneralizedCE1(nn.Module):
    def __init__(self):
        super(GeneralizedCE1, self).__init__()

    def forward(self, logits, label, ut):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        q = 0.5

        # first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**ut)/ut) * label, dim=1) / pos_factor)
        a = 0.2
        first_term = torch.mean(torch.sum(torch.exp(-ut) *((1 - (logits + 1e-7) ** q) / q) * label, dim=1) / pos_factor) + a * torch.mean(ut)  #
        # second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**(1-ut))/(1-ut)) * (1-label), dim=1) / neg_factor)
        second_term = torch.mean(torch.sum(torch.exp(-ut) *((1 - (1 - logits + 1e-7) ** (1 - q)) / (1 - q)) * (1 - label), dim=1) / neg_factor) + a * torch.mean(ut)  # torch.exp(-ut) *
        # second_term = 0
        # ((1 - (1 - logits + 1e-7)**self.q)/self.q)

        # print(first_term, second_term)
        return first_term + second_term


class MeanClusteringError(nn.Module):
    """
    Mean Absolute Error
    """

    def __init__(self, num_classes, tau=1):
        super(MeanClusteringError, self).__init__()
        # self.register_buffer('embedding', torch.eye(num_classes))
        self.tau = tau

    def to_onehot(self, target):
        return self.embedding[target]

    def forward(self, input, target, threshold=1):
        pred = F.softmax(input / self.tau, dim=1)
        q = target.permute(0,2,1).reshape(-1,1).detach()                                 # [100, 10]  one-hot标签， target是数字标签
        q_ = torch.cat((q,1-q),dim=1)

        # print(pred)
        # exit()
        p = ((1. - q_) * pred).sum(1) / pred.sum(1)
        return (p.log()).mean()


class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):                #　　0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss