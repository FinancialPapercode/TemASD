from .gnn import GNNEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
def cos_sim(x1, x2, tau: float, norm: bool = False):
    device = x1.device  # 获取 x1 所在的设备
    x2 = x2.to(device)  # 将 x2 移动到与 x1 相同的设备上
    if norm:
        return F.softmax(x1 @ x2.T / tau, dim=1)
    else:
        return torch.exp(x1 @ x2.T / tau)

def RBF_sim(x1, x2, tau: float, norm: bool = False):
    device = x1.device  # 获取 x1 所在的设备
    x2 = x2.to(device)  # 将 x2 移动到与 x1 相同的设备上
    xx1, xx2 = torch.stack(len(x2) * [x1]), torch.stack(len(x1) * [x2])
    sub = xx1.transpose(0, 1) - xx2
    R = (sub * sub).sum(dim=2)
    if norm:
        return F.softmax(-R / (tau * tau), dim=1)
    else:
        return torch.exp(-R / (tau * tau))

class AbnormalSubgraphOrderEmbedding(nn.Module):
    def __init__(self, args):
        super(AbnormalSubgraphOrderEmbedding, self).__init__()

        self.encoder = GNNEncoder(args)
        self.margin = args.margin
        self.device = args.device
        self.center = nn.Parameter(torch.randn(args.pairs_size, args.hidden_dim, dtype=torch.float32))
        self.tau = 0.4
        self.lamda = 1.0
        self.args = args
        self.similarity = 'cos'

    def node_contrast(self,h1,h2):
        # compute similarity
        s12 = cos_sim(h1,h2,self.tau,norm=False)
        s21 = s12

        # compute InfoNCE
        loss12 = -torch.log(s12.diag())+torch.log(s12.sum(1))
        loss21 = -torch.log(s21.diag())+torch.log(s21.sum(1))
        L_node = (loss12+loss21)/2

        return L_node.mean()
    def community_assign(self,h):
        if self.similarity=='cos':
            return cos_sim(h.detach(),F.normalize(self.center),self.tau,norm=True)
        return RBF_sim(h.detach(),F.normalize(self.center),self.tau,norm=True)

    def DeCA(self, R, edge_index):
        n = len(R)
        m = n * (n - 1)

        # Ensure edge_index is of type long
        edge_index = edge_index.long()

        # Create the adjacency matrix
        A = torch.zeros(n, n, device=R.device, dtype=torch.float32)
        A[edge_index[0], edge_index[1]] = 1

        # Edge density constraint
        DF = R.T @ A @ R
        return (self.lamda * DF.sum() - (n - 1 + self.lamda) * DF.trace()) / m + (R.T @ R).trace() / n / 2

    def community_contrast(self, h1, h2, R1, R2):
        device = h1.device  # 确保所有张量在同一设备上
        # gather communities
        index1, index2 = R1.detach().argmax(dim=1), R2.detach().argmax(dim=1)
        C1, C2 = [], []
        for i in range(self.args.pairs_size):
            h_c1, h_c2 = h1[index1 == i], h2[index2 == i]
            if h_c1.shape[0] > 0:
                C1.append(h_c1.sum(dim=0) / h_c1.shape[0])
            else:
                C1.append(torch.zeros(h1.shape[1], device=device, dtype=torch.float32))
            if h_c2.shape[0] > 0:
                C2.append(h_c2.sum(dim=0) / h_c2.shape[0])
            else:
                C2.append(torch.zeros(h2.shape[1], device=device, dtype=torch.float32))
        C1, C2 = torch.stack(C1), torch.stack(C2)

        # compute similarity
        if self.similarity == 'cos':
            s_h1_c2 = cos_sim(h1, C2.detach(), self.tau, norm=False)
            s_h2_c1 = cos_sim(h2, C1.detach(), self.tau, norm=False)
            ws_h1_c2 = s_h1_c2
            ws_h2_c1 = s_h2_c1
        else:
            s_h1_c2 = RBF_sim(h1, C2.detach(), self.tau, norm=False)
            s_h2_c1 = RBF_sim(h2, C1.detach(), self.tau, norm=False)
            h1_extend, h2_extend = torch.stack([C1.detach()] * len(C1)), torch.stack([C2.detach()] * len(C2))
            h1_sub, h2_sub = h1_extend - h1_extend.transpose(0, 1), h2_extend - h2_extend.transpose(0, 1)
            w1, w2 = torch.exp(-self.gamma * (h1_sub * h1_sub).sum(2)), torch.exp(
                -self.gamma * (h2_sub * h2_sub).sum(2))
            ws_h1_c2 = s_h1_c2 * w2[index2]
            ws_h2_c1 = s_h2_c1 * w1[index1]

        # node-community contrast
        self_s12 = s_h1_c2.gather(1, index2.unsqueeze(-1)).squeeze(-1)
        self_s21 = s_h2_c1.gather(1, index1.unsqueeze(-1)).squeeze(-1)
        loss12 = -torch.log(self_s12) + torch.log(
            self_s12 +
            ws_h1_c2.sum(1) - ws_h1_c2.gather(1, index2.unsqueeze(-1)).squeeze(-1)
        )
        loss21 = -torch.log(self_s21) + torch.log(
            self_s21 +
            ws_h2_c1.sum(1) - ws_h2_c1.gather(1, index1.unsqueeze(-1)).squeeze(-1)
        )
        L_community = (loss12 + loss21) / 2

        return L_community.mean()

    def align_nested_lists(self,list1, list2):
        max_length = max(len(max(list1, key=len)), len(max(list2, key=len)))
        aligned_list1 = [sublist + [0] * (max_length - len(sublist)) for sublist in list1]
        aligned_list2 = [sublist + [0] * (max_length - len(sublist)) for sublist in list2]
        return aligned_list1, aligned_list2

    def calculate_kl_divergence_loss(self,list1, list2):
        list1_aligned, list2_aligned = self.align_nested_lists(list1, list2)
        kl_div_losses = []
        for sublist1, sublist2 in zip(list1_aligned, list2_aligned):
            tensor1 = torch.tensor(sublist1, dtype=torch.float32)
            tensor2 = torch.tensor(sublist2, dtype=torch.float32)
            kl_div_loss = F.kl_div(F.log_softmax(tensor1, dim=-1),F.softmax(tensor2, dim=-1),reduction='batchmean')
            kl_div_losses.append(kl_div_loss.item())
        return kl_div_losses

    def align_lists(self,list1, list2, fill_value=0):
        max_length = max(len(lst) for lst in list1 + list2)

        for lst in list1:
            while len(lst) < max_length:
                lst.append(fill_value)

        for lst in list2:
            while len(lst) < max_length:
                lst.append(fill_value)

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_bs - emb_as) ** 2, dim=1)
        return e

    def criterion(self, pred, labels, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b):
        emb_as, emb_bs = pred
        device = self.device  # 获取设备
        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=device), emb_bs - emb_as) ** 2, dim=1)
        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0, device=device), margin - e)[labels == 0]
        kl_div_loss = F.kl_div(F.log_softmax(emb_as, dim=1), F.softmax(emb_bs, dim=1), reduction='batchmean')

        self.align_lists(pos_benford_a,pos_benford_b)
        posa_tensor = torch.tensor(pos_benford_a, dtype=torch.float).to(device)
        posb_tensor = torch.tensor(pos_benford_a, dtype=torch.float).to(device)
        kl_div_loss_pos = F.kl_div(F.log_softmax(posa_tensor, dim=1), F.softmax(posb_tensor, dim=1), reduction='batchmean')
        self.align_lists(neg_benford_a,neg_benford_b)
        nega_tensor = torch.tensor(neg_benford_a, dtype=torch.float).to(device)
        negb_tensor = torch.tensor(neg_benford_b, dtype=torch.float).to(device)
        kl_div_loss_neg = F.kl_div(F.log_softmax(nega_tensor, dim=1), F.softmax(negb_tensor, dim=1), reduction='batchmean')


        return torch.sum(e) + kl_div_loss_pos +kl_div_loss_neg

