import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from .policy_net import MLP, GCN
from .symbol import EXPAND, EXCLUDE, VIRTUAL_EXCLUDE_NODE, VIRTUAL_EXPAND_NODE


class RefinementAgent:
    def __init__(self, args):
        self.args = args
        self.exclude_net, self.expand_net, self.gcn = self.network_init()
        self.exclude_opt = optim.Adam(self.exclude_net.parameters(), lr=args.agent_lr)
        self.expand_opt = optim.Adam(self.expand_net.parameters(), lr=args.agent_lr)

    def network_init(self):
        exclude_net = MLP(65, 32, 1)
        expand_net = MLP(65, 32, 1)
        gcn = GCN()
        return exclude_net, expand_net, gcn

    def learn(self, log_prob, reward, cal_type=EXCLUDE):
        loss = (-log_prob * reward).sum()
        if cal_type == EXCLUDE:
            self.exclude_opt.zero_grad()
            loss.backward()
            self.exclude_opt.step()
        else:
            self.expand_opt.zero_grad()
            loss.backward()
            self.expand_opt.step()

    def choose_action(self, subgraph_obj, cal_type=EXCLUDE):
        idx_list = []
        if cal_type == EXCLUDE:
            idx_list = sorted([idx for idx in subgraph_obj.mapping.keys() if subgraph_obj.mapping[idx] in subgraph_obj.pred_subgraphs])
            if len(idx_list) <= 1:
                return None
        elif cal_type == EXPAND:
            expander = sorted([node for node in subgraph_obj.nodes if node not in subgraph_obj.pred_subgraphs])
            if len(expander) <= 1 or not subgraph_obj.expand:
                return None
            idx_list = sorted([idx for idx in subgraph_obj.mapping.keys() if subgraph_obj.mapping[idx] in expander])

        tmp_mapping = {idx: subgraph_obj.mapping[idx_val] for idx, idx_val in enumerate(idx_list)}
        feat = subgraph_obj.feat_mat[idx_list, :]

        if cal_type == EXCLUDE:
            score = self.exclude_net(torch.FloatTensor(feat))
        else:
            score = self.expand_net(torch.FloatTensor(feat))
        probs = F.softmax(torch.FloatTensor(np.squeeze(score)), dim=-1)
        if torch.isnan(probs).any():
            print("Warning: probs contains NaN values. Returning None.")
            return None
        action_dist = Categorical(probs)
        action = action_dist.sample()  # node idx
        log_prob = action_dist.log_prob(action)

        node = tmp_mapping[action.item()]

        if (cal_type == EXCLUDE and node == VIRTUAL_EXCLUDE_NODE) or \
                (cal_type == EXPAND and node == VIRTUAL_EXPAND_NODE):
            return None
        return {
            "log_prob": log_prob,
            "node": node
        }
