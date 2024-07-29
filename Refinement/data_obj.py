import random
import networkx as nx
import numpy as np
import torch
from .symbol import VIRTUAL_EXCLUDE_NODE, VIRTUAL_EXPAND_NODE
from utils import generate_ego_net ,generate_outer_boundary_3,generate_outer_boundary_with_dynamic_threshold
import math,collections
import scipy.stats as stats

class AbnormalSubgraph:
    def __init__(self, feat_mat, pred_subgraphs, true_subgraphs, nodes, subgraph, mapping, expand=True):
        """
        :param feat_mat: node feature matrix
        :param pred_subgraphs: init predicted AbnormalSubgraphs
        :param true_subgraphs: corresponding ground-truth AbnormalSubgraphs
        :param nodes: nodes set ( pred_subgraphs + outer_boundary )
        :param subgraph: `nx.TGraph` object
        :param mapping:
        :param expand:
        """
        self.nodes = nodes
        self.feat_mat = feat_mat
        self.pred_subgraphs = pred_subgraphs
        self.true_subgraphs = true_subgraphs
        self.graph = subgraph
        self.mapping = mapping
        self.expand = expand

        # Virtual node for stopping exclusion
        self.nodes.append(VIRTUAL_EXCLUDE_NODE)
        self.pred_subgraphs.append(VIRTUAL_EXCLUDE_NODE)
        self.mapping[len(self.nodes)-1] = VIRTUAL_EXCLUDE_NODE

        # Virtual node for stopping expansion
        self.nodes.append(VIRTUAL_EXPAND_NODE)
        self.mapping[len(self.nodes)-1] = VIRTUAL_EXPAND_NODE

        # Virtual nodes embedding (all zero)
        self.feat_mat = np.vstack((self.feat_mat, np.zeros((2, 64))))

        # Augment node embedding with POSITION-FLAG
        position_flag = self.generate_position_flag()
        self.feat_mat = np.hstack((position_flag, self.feat_mat))

    def generate_position_flag(self):
        result = np.zeros((self.feat_mat.shape[0], 1))
        for idx, node in self.mapping.items():
            if node in self.pred_subgraphs:
                result[idx, 0] = 1
        return result

    def get_start_digit(self,v):
        if v == 0:
            return 0
        if v < 0:
            v = -v
        while v < 1:
            v = v * 10
        return int(str(v)[:1])

    def compute_cost(self, graph,nodes,benford_G,choice="f1"):

        benford = []
        et = 0
        for i in range(9):
            benford.append(math.log10(1 + 1 / (i + 1)))
        subgraph = benford_G.subgraph(nodes)
        t = nx.get_edge_attributes(subgraph, 'weight')
        weights = 0
        el = []
        subgraph = nx.convert_node_labels_to_integers(subgraph)
        for e in subgraph.edges():
            for w in subgraph[e[0]][e[1]]['weight']:
                el.append([e[0], e[1], self.get_start_digit(w)])

        res = []
        for j in t:
            res += t[j]
        xx = collections.Counter(res)
        obs = []
        for i in range(1, 10):
            if i not in xx:
                obs.append(0)
            else:
                obs.append(xx[i])
        times = sum(obs)
        c3, p3 = stats.chisquare(obs, np.array(benford) * times)
        for i in range(subgraph.number_of_nodes()):
            tmp_deg = sum([len(subgraph[i][j]['weight']) for j in subgraph[i]])
            et += tmp_deg
        result = self.calculate_ratio(c3,et,subgraph)
        return  result*10


    def calculate_ratio(self,c3, et, subgraph):
        if subgraph.number_of_nodes() == 0:
            return 0

        phi = c3 / subgraph.number_of_nodes()
        density = et / (2 * subgraph.number_of_nodes())

        if math.isnan(phi) or math.isinf(phi) or math.isnan(density) or math.isinf(density):
            return 0
        return phi / density

    def calculate_ratio2(self,c3, et, subgraph):
        if subgraph.number_of_nodes() == 0:
            return 0

        phi = c3 / subgraph.number_of_nodes()

        if math.isnan(phi) or math.isinf(phi) :
            return 0
        return phi


    def step(self, graph,emb_updater):
        """Move into the next **State** with `emb_updater`"""
        edges = list(graph.subgraph(self.pred_subgraphs).edges())
        litlegraph = graph.subgraph(self.pred_subgraphs)
        e_tensor = torch.zeros((2, len(edges)), dtype=int)
        revert_mapping = {node: idx for idx, node in self.mapping.items()}
        edge_weight = []
        i=0
        for u, v, data in litlegraph.edges(data=True):
            weight = data.get('weight')

            e_tensor[0][i], e_tensor[1][i] = revert_mapping[u], revert_mapping[v]
            edge_weight.append(weight)
            i += 1
        # Update Node Embedding
        x_tensor = torch.FloatTensor(self.feat_mat)
        new_x = emb_updater(x_tensor, e_tensor, edge_weight)

        position_flag = self.generate_position_flag()
        new_feat_mat = np.hstack((position_flag, new_x.detach().numpy()))
        return new_feat_mat

    def apply_exclude(self, node, cost_choice,graph,nodes,benford_G):
        """Apply Exclude action and return corresponding Reward"""
        # print("nodes:"+str(nodes))
        pre_cost = self.compute_cost(graph,nodes,benford_G,choice=cost_choice)
        # print("pred_subgraphs:" + str(self.pred_subgraphs))
        if node in self.pred_subgraphs:
            if len(self.pred_subgraphs)>=3:
                self.pred_subgraphs.remove(node)
                sub = graph.subgraph(self.pred_subgraphs)
                if not nx.is_connected(sub):
                    self.pred_subgraphs.append(node)
        return self.compute_cost(graph, self.pred_subgraphs, benford_G, choice=cost_choice) - pre_cost

    def apply_expand(self, node, cost_choice,graph,nodes,benford_G):
        """Apply Expand action and return corresponding Reward"""
        pre_cost = self.compute_cost(graph,nodes,benford_G,choice=cost_choice)
        if node not in self.pred_subgraphs:
            self.pred_subgraphs.append(node)
        return self.compute_cost(graph, self.pred_subgraphs, benford_G, choice=cost_choice) - pre_cost


class DataProcessor(object):
    def __init__(self, args, dataset, feat_mat, graph, train_abnormalsubgraphs, valid_abnormalsubgraphs, node_chisquares,Tgraph):
        self.args = args
        self.dataset = dataset
        self.graph = graph
        self.feat_mat = feat_mat
        self.train_abnormalsubgraphs, self.valid_abnormalsubgraphs = train_abnormalsubgraphs, valid_abnormalsubgraphs
        self.node_chisquares = node_chisquares
        self.Tgraph = Tgraph

    def generate_data(self, batch_size=64, valid=False):
        subgraphs = self.train_abnormalsubgraphs if not valid or len(self.valid_abnormalsubgraphs) == 0 else self.valid_abnormalsubgraphs

        train_set = []
        total_nodes = []

        for _ in range(batch_size):
            true_subgraphs = random.choice(subgraphs)



            subgraph = self.graph.subgraph(true_subgraphs)
            nodes = sorted(list(subgraph.nodes()))
            degree = {node: sum(self.graph[node][neighbor]['weight'] for neighbor in subgraph.neighbors(node)) for node
                      in subgraph.nodes()}
            node2degree = {node: degree[node] for node in nodes}
            degree = np.array(list(node2degree.values()))
            degree = degree / np.sum(degree, axis=0)
            root_node = np.random.choice(nodes, p=degree.ravel())


            ego_net = generate_ego_net(self.graph, root_node, _,self.Tgraph,k=self.args.kego, max_size=self.args.ab_size, choice="neighbors")

            if valid:
                train_set.append(ego_net)
                continue

            # outer_nodes = generate_outer_boundary_3(self.graph, ego_net, max_size=10)
            outer_nodes = generate_outer_boundary_with_dynamic_threshold(self.graph, ego_net, self.Tgraph,max_size=10)
            expand = False if len(outer_nodes) == 0 else True

            all_nodes = sorted(ego_net + outer_nodes)

            feat_mat = self.feat_mat[all_nodes, :]
            mapping = {idx: node for idx, node in enumerate(all_nodes)}

            subgraph_obj = AbnormalSubgraph(feat_mat, ego_net, true_subgraphs, all_nodes, self.graph.subgraph(all_nodes), mapping,
                                expand=expand)

            train_set.append(subgraph_obj)
            total_nodes.append(ego_net)
        return train_set,total_nodes
