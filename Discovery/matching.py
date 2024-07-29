import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.stats as stats
from .model import AbnormalSubgraphOrderEmbedding
import random
from utils import sample_neigh, batch2graphs, generate_embedding, generate_ego_net,batch2graphstimeslice,batch2graphstimeslice_nsubgraph
import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from scipy.special import kl_div
import math
from math import exp
import collections
import time
import multiprocessing
import torch.nn.functional as F
from utils import load, feature_augmentation, split_abnormalsubgraphs,eval_scores_timeporal, eval_scores, load_benford,load_syn, loadTime

class Projection(nn.Sequential):
    def __init__(self, num_hidden, num_proj_hidden):
        super(Projection, self).__init__(
            nn.Linear(num_hidden, num_proj_hidden),
            nn.ELU(),
            nn.Linear(num_proj_hidden, num_hidden)
        )

    def forward(self, x):
        device = next(self.parameters()).device  # 获取模型权重所在的设备
        x = x.to(device)  # 将输入张量移动到同一设备上
        x = super(Projection, self).forward(x)
        return F.normalize(x)

class AbnormalMatching:
    def __init__(self, args, graph, train_abnormalsubgraphs, val_abnormalsubgraphs, edges, edgelist, benford_G, node_chisquares,Tgraph):
        self.args = args

        self.graph = graph
        self.edges = edges
        self.seen_nodes = {node for com in train_abnormalsubgraphs + val_abnormalsubgraphs for node in com}
        self.train_abnormalsubgraphs, self.val_abnormalsubgraphs = self.init_subgraphs(train_abnormalsubgraphs), self.init_subgraphs(val_abnormalsubgraphs)
        self.edgelist= edgelist
        self.benford_G = benford_G
        self.node_chisquares = node_chisquares
        self.model = self.load_model()
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.writer = SummaryWriter(args.writer_dir)
        self.Tgraph = Tgraph
        self.proj = Projection(self.args.hidden_dim, self.args.hidden_dim)

    def load_model(self, load=False):
        """Load AbnormalSubgraphOrderEmbedding"""
        model = AbnormalSubgraphOrderEmbedding(self.args)
        model.to(self.args.device)
        if self.args.subm_path and load:
            model.load_state_dict(torch.load(self.args.subm_path, map_location=self.args.device))
        return model

    def init_subgraphs(self, subs):
        if len(subs) > 0:
            return [self.graph.subgraph(com) for com in subs if len(list(self.graph.subgraph(com).edges())) > 0]
        return []

    def find_weights(self, edge_list, other_variable):
        edge_index = {}

        # 构建边索引
        for item in other_variable:
            edge_index[(item[0], item[1])] = item[2]
            edge_index[(item[1], item[0])] = item[2]

        weights = []
        edge_info = []
        for edge in edge_list:
            if edge in edge_index:
                weight = edge_index[edge]
                weights.append(weight)
                edge_info.append((edge[0], edge[1], weight))

        return weights, edge_info

    def generate_batch(self, batch_size, benford_G,TGraph,valid=False, min_size=2, max_size=100):
        graphs = self.train_abnormalsubgraphs if not valid or len(self.val_abnormalsubgraphs) == 0 else self.val_abnormalsubgraphs

        pos_a, pos_b = [], []
        pos_a_nodelist,pos_b_nodelist = [],[]

        ratio = self.args.fine_ratio

        pos_edge_a, pos_edge_b = [], []

        pos_benford_a,pos_benford_b = [], []

        # Generate positive pairs
        for i in range(batch_size // 2):
            prob = random.random()
            if prob <= ratio:
                # Fine-grained sampling
                size = random.randint(min_size + 1, max_size)
                graph, a, _ = sample_neigh(graphs, size, self.edges, graph_weight)
                if len(a) - 1 <= min_size:
                    b = a
                else:
                    b = a[:random.randint(max(len(a) - 2, min_size), len(a))]
            else:
                graph = None
                while graph is None or len(graph) < min_size:
                    graph = random.choice(graphs)
                a = graph.nodes
                time_graph = TGraph.subgraph(a)

                graph_weight = 0
                edge_info = []
                chi_info = []

                for edge in graph.edges(data=True):
                    weight = edge[2]['weight']
                    graph_weight += weight
                    chi = int(100*math.sqrt(self.node_chisquares[edge[0]]*self.node_chisquares[edge[1]]))
                    edge_info.append([edge[0], edge[1], edge[2]['weight']])
                    chi_info.append([edge[0],edge[1],chi])

                if graph_weight<=2:
                    continue
                _, b, updated_edge_info = sample_neigh([graph],
                                                       random.randint(max(graph_weight - 2, min_size), graph_weight),
                                                       self.edges, graph_weight)
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)


            neighaedge = []
            neigh_a_new = nx.Graph(neigh_a)
            neigh_a_new.add_weighted_edges_from(edge_info, weight='weight')
            for u,v,data in neigh_a_new.edges(data = True):
                neighaedge.append(data['weight'])

            neighbedge = []
            neigh_b_new = nx.Graph(neigh_b)
            neigh_b_new.add_weighted_edges_from(updated_edge_info, weight='weight')
            for u,v,data in neigh_b_new.edges(data = True):
                neighbedge.append(data['weight'])

            neighaedgebenford = []
            agraph = benford_G.subgraph(a)
            for u,v,data in agraph.edges(data = True):
                for num in data['weight']:
                    neighaedgebenford.append(num)

            neighbedgebenford = []
            bgraph = benford_G.subgraph(b)
            for u, v, data in bgraph.edges(data=True):
                for num in data['weight']:
                    neighbedgebenford.append(num)

            if len(neigh_a_new.edges()) > 0 and len(neigh_b_new.edges()) > 0:
                pos_a.append(neigh_a_new)
                pos_b.append(neigh_b_new)
                pos_a_nodelist.append(neigh_a_new.nodes())
                pos_b_nodelist.append(neigh_b_new.nodes())
                pos_edge_a.append(neighaedge)
                pos_edge_b.append(neighbedge)
                pos_benford_a.append(neighaedgebenford)
                pos_benford_b.append(neighbedgebenford)

        # Generate negative pairs
        neg_a, neg_b = [], []
        neg_edge_a, neg_edge_b = [], []
        neg_benford_a,neg_benford_b = [], []
        neg_a_nodelist , neg_b_nodelist = [],[]
        for i in range(batch_size // 2):
            prob = random.random()
            if prob <= ratio:
                size = random.randint(min_size + 1, max_size)
                graph_a, a, _ = sample_neigh(graphs, random.randint(min_size, size), self.edges, graph_weight)
                graph_b, b, _ = sample_neigh(graphs, size, self.edges, graph_weight)
            else:
                graph_b = None
                while graph_b is None or len(graph_b) < min_size:
                    graph_b = random.choice(graphs)
                b = graph_b.nodes

                graph_weight = 0
                edge_info_neg = []
                for edge in graph_b.edges(data=True):
                    weight = edge[2]['weight']
                    graph_weight += weight
                    edge_info_neg.append([edge[0], edge[1], edge[2]['weight']])

                if graph_weight<=2:
                    continue

                graph_a, a, updated_edge_info_neg = sample_neigh(graphs, random.randint(min_size, graph_weight),
                                                                 self.edges, graph_weight)
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)

            neighaedge = []
            neigh_a_new = nx.Graph(neigh_a)
            neigh_a_new.add_weighted_edges_from(updated_edge_info_neg, weight='weight')
            for u,v,data in neigh_a_new.edges(data = True):
                neighaedge.append(data['weight'])

            neighbedge = []
            neigh_b_new = nx.Graph(neigh_b)
            neigh_b_new.add_weighted_edges_from(edge_info_neg, weight='weight')
            for u,v,data in neigh_b_new.edges(data = True):
                neighbedge.append(data['weight'])

            neighaedgebenford = []
            agraph = benford_G.subgraph(a)
            for u,v,data in agraph.edges(data = True):
                for num in data['weight']:
                    neighaedgebenford.append(num)

            neighbedgebenford = []
            bgraph = benford_G.subgraph(b)
            for u, v, data in bgraph.edges(data=True):
                for num in data['weight']:
                    neighbedgebenford.append(num)

            if len(neigh_a_new.edges()) > 0 and len(neigh_b_new.edges()) > 0:
                neg_a.append(neigh_a_new)
                neg_b.append(neigh_b_new)
                neg_a_nodelist.append(neigh_a_new.nodes())
                neg_b_nodelist.append(neigh_b_new.nodes())
                neg_edge_a.append(neighaedge)
                neg_edge_b.append(neighbedge)
                neg_benford_a.append(neighaedgebenford)
                neg_benford_b.append(neighbedgebenford)

        pos_a = batch2graphstimeslice(pos_a,TGraph,48,device=self.args.device)
        pos_b = batch2graphstimeslice(pos_b,TGraph,48,device=self.args.device)
        neg_a = batch2graphstimeslice(neg_a,TGraph, 48,device=self.args.device)
        neg_b = batch2graphstimeslice(neg_b, TGraph,48,device=self.args.device)
        return pos_a, pos_b, neg_a, neg_b,pos_edge_a,pos_edge_b,neg_edge_a,neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b,pos_a_nodelist,pos_b_nodelist,neg_a_nodelist,neg_b_nodelist

    def train_epoch(self, epochs):
        self.model.share_memory()

        batch_size = self.args.batch_size
        pairs_size = self.args.pairs_size
        device = self.args.device

        valid_set = []
        for _ in range(batch_size):
            pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b,pos_a_nodelist,pos_b_nodelist,neg_a_nodelist,neg_b_nodelist = self.generate_batch(pairs_size, self.benford_G,self.Tgraph,valid=True)
            valid_set.append((pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b))

        for epoch in range(epochs):
            for batch in range(batch_size):
                self.model.train()

                pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b,pos_a_nodelist,pos_b_nodelist,neg_a_nodelist,neg_b_nodelist = self.generate_batch(pairs_size,self.benford_G,self.Tgraph,)
                # Get embeddings
                emb_pos_a, emb_pos_b = self.model.encoder(pos_a), self.model.encoder(pos_b)
                emb_neg_a, emb_neg_b = self.model.encoder(neg_a), self.model.encoder(neg_b)

                max_len = max(emb_pos_a.size(0), emb_pos_b.size(0), emb_neg_a.size(0), emb_neg_b.size(0))
                padding_value = 0
                emb_pos_a = F.pad(emb_pos_a, (0, 0, 0, max_len - emb_pos_a.size(0)), value=padding_value)
                emb_pos_b = F.pad(emb_pos_b, (0, 0, 0, max_len - emb_pos_b.size(0)), value=padding_value)
                emb_neg_a = F.pad(emb_neg_a, (0, 0, 0, max_len - emb_neg_a.size(0)), value=padding_value)
                emb_neg_b = F.pad(emb_neg_b, (0, 0, 0, max_len - emb_neg_b.size(0)), value=padding_value)

                mask_pos_a = torch.ones_like(emb_pos_a)
                mask_pos_a[emb_pos_a == padding_value] = 0
                mask_pos_b = torch.ones_like(emb_pos_b)
                mask_pos_b[emb_pos_b == padding_value] = 0
                mask_neg_a = torch.ones_like(emb_neg_a)
                mask_neg_a[emb_neg_a == padding_value] = 0
                mask_neg_b = torch.ones_like(emb_neg_b)
                mask_neg_b[emb_neg_b == padding_value] = 0

                # 拼接嵌入向量
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
                mask_a = torch.cat((mask_pos_a, mask_neg_a), dim=0)
                mask_b = torch.cat((mask_pos_b, mask_neg_b), dim=0)

                labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(device)
                pred = self.model(emb_as, emb_bs)
                
                self.model.zero_grad()
                loss = self.model.loss_cl_mask(emb_as, emb_bs, pos_benford_a, pos_benford_b, neg_benford_a, neg_benford_b, mask_a, mask_b)
                loss = torch.sum(loss)
                loss.backward()
                self.opt.step()

                if (batch + 1) % 5 == 0:
                    self.writer.add_scalar(f"subgraphM Loss/Train", loss.item(), batch + epoch * batch_size)
                    print(f"Epoch {epoch + 1}, Batch{batch + 1}, Loss {loss.item():.4f}")
                if (batch + 1) % 10 == 0:
                    self.valid_model(valid_set, batch + epoch * batch_size)
        torch.save(self.model.state_dict(), self.args.writer_dir + "/subgraphrm.pt")

    def valid_model(self, valid_set, batch_num):
        """Test model on `valid_set`"""
        self.model.eval()
        device = self.args.device

        total_loss = 0
        for pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b in valid_set:
            labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(device)

            with torch.no_grad():
                emb_pos_a, emb_pos_b = self.model.encoder(pos_a), self.model.encoder(pos_b)
                emb_neg_a, emb_neg_b = self.model.encoder(neg_a), self.model.encoder(neg_b)
                
                max_len = max(emb_pos_a.size(0), emb_pos_b.size(0), emb_neg_a.size(0), emb_neg_b.size(0))
                padding_value = 0
                emb_pos_a = F.pad(emb_pos_a, (0, 0, 0, max_len - emb_pos_a.size(0)), value=padding_value)
                emb_pos_b = F.pad(emb_pos_b, (0, 0, 0, max_len - emb_pos_b.size(0)), value=padding_value)
                emb_neg_a = F.pad(emb_neg_a, (0, 0, 0, max_len - emb_neg_a.size(0)), value=padding_value)
                emb_neg_b = F.pad(emb_neg_b, (0, 0, 0, max_len - emb_neg_b.size(0)), value=padding_value)

                mask_pos_a = torch.ones_like(emb_pos_a)
                mask_pos_a[emb_pos_a == padding_value] = 0
                mask_pos_b = torch.ones_like(emb_pos_b)
                mask_pos_b[emb_pos_b == padding_value] = 0
                mask_neg_a = torch.ones_like(emb_neg_a)
                mask_neg_a[emb_neg_a == padding_value] = 0
                mask_neg_b = torch.ones_like(emb_neg_b)
                mask_neg_b[emb_neg_b == padding_value] = 0

                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
                mask_a = torch.cat((mask_pos_a, mask_neg_a), dim=0)
                mask_b = torch.cat((mask_pos_b, mask_neg_b), dim=0)

                pred = self.model(emb_as, emb_bs)
                
                loss = self.model.loss_cl_mask(emb_as, emb_bs, pos_benford_a, pos_benford_b, neg_benford_a,
                                                neg_benford_b, mask_a, mask_b)
                total_loss += loss.item()
        total_loss /= len(valid_set)
        self.writer.add_scalar(f"subgraphM Loss/Val", loss.item(), batch_num)
        print("[Eval-Test] Validation Loss{:.4f}".format(total_loss))


    def get_start_digit(self,v):
        if v == 0:
            return 0
        if v < 0:
            v = -v
        while v < 1:
            v = v * 10
        return int(str(v)[:1])

    def kl_divergence(self,p, q):
        return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if p[i] > 0)

    def count_occ(self,l, dist_len, adjust_idx=0):
        result = [0 for i in range(dist_len)]
        for e in l:
            result[e - adjust_idx] += 1
        return result

    def node_chisquare(self,edgelistl, node_num, dist, adjust_idx=1, directed='both', diff_func='chi'):
        '''
        edgelist: list of edges. Each item is a tuple contains (node_from, node_to, edge_weight) representing a weighted edge.
        node_num: number of nodes.
        dist: list of float sum to 1, describe the distribution used to calculate the chi square statistic.
        '''
        node_induced_dist = [[] for i in range(node_num)]
        node_chis = []
        for edge in edgelistl:
            if directed == 'both' or directed == 'out':
                node_induced_dist[edge[0]].append(edge[2])
            if directed == 'both' or directed == 'in':
                node_induced_dist[edge[1]].append(edge[2])
        #             G[node][neighbor]['weight']
        for node_dist in node_induced_dist:
            count_dist = self.count_occ(node_dist, len(dist), adjust_idx)
            #        get the chi square statistic, the higher it is, more abnormal the node is
            if diff_func == 'chi':
                node_chis.append(stats.chisquare(count_dist, sum(count_dist) * np.array(dist))[0])
            elif diff_func == 'kl':
                node_chis.append(self.kl_divergence(np.array(count_dist) / sum(count_dist), np.array(dist)))
        return node_chis

    def generate_ego_net_wrapper(self, graph, sublist, results, benford_G, Tgraph,lock):
        graphs = [generate_ego_net(graph, g, benford_G, Tgraph, k=self.args.kego, max_size=self.args.ab_size,
                                        choice="multi") for g in sublist]
        # results.extend(graphs)
        with lock:
            results.extend(graphs)

    def load(self,name):
        """Load snap dataset"""
        abnormalsubgraphs = open(f"./dataset/jan18-hour-weight/jan18-hour-weight-1.90.anomaly.txt")
        edges = open(f"./dataset/jan18-hour-weight/{name}-weight.txt")

        abnormalsubgraphs = [[int(i) for i in x.split()] for x in abnormalsubgraphs]
        edges = [[int(i) for i in e.split()] for e in edges]

        nodes = set()
        edge_weights = []
        for u, v, w in edges:
            nodes.add(u)
            nodes.add(v)
            edge_weights.append(w)

        print(f"[{name.upper()}], #Nodes {len(nodes)}, #Edges {len(edges)} #AbnormalSubgraphs {len(abnormalsubgraphs)}")
        nodes = set(nodes)

        return nodes, edges, abnormalsubgraphs

    def loadTime(self,name):
        """Load snap dataset"""
        with open(f"./dataset/jan18-hour-weight/{name}.txt", 'r') as file:
            edges = [[int(i) for i in line.split()] for line in file]

        nodes = set()
        edge_weights = []

        for u, v, timestamp in edges:
            nodes.add(u)
            nodes.add(v)
            edge_weights.append(timestamp)


        print(f"[{name.upper()}], #TNodes {len(nodes)}, #TEdges {len(edges)} ")

        return nodes, edges
    

    def load_embedding(self):
        query_emb = generate_embedding(self.train_abnormalsubgraphs + self.val_abnormalsubgraphs, self.model, self.Tgraph,device=self.args.device)
        k = 50
        n_node = len(list(self.graph.nodes()))
        batch_size = 30000
        batch_len = int((n_node / batch_size) + 1)
        benford = []
        for i in range(9):
            benford.append(math.log10(1 + 1 / (i + 1)))

        all_emb = np.zeros((n_node, self.args.output_dim))
        for hour in range(1, 10):
            nodes_1hour, edges_1hour, abnormalsubgraphs = self.load(str(hour))
            graph_1hour, _ = feature_augmentation(nodes_1hour, edges_1hour)

            Tgraph_1hour = nx.Graph()
            tnodes_1hour, tedges_1hour = self.loadTime(str(hour))
            for (u, v, weight) in tedges_1hour:
                if Tgraph_1hour.has_edge(u, v):
                    current_weight = Tgraph_1hour[u][v]['weight']
                    if isinstance(current_weight, list):
                        current_weight.append(weight)
                    else:
                        Tgraph_1hour[u][v]['weight'] = [current_weight, weight]
                else:
                    Tgraph_1hour.add_edge(u, v, weight=[weight])
            n_node = len(list(graph_1hour.nodes()))
            batch_len = int((n_node / batch_size) + 1)
            for batch_num in range(batch_len):
                start_time_ego = time.time()
                start, end = batch_num * batch_size, min((batch_num + 1) * batch_size, n_node)


                num_iterations = end-start
                num_processes = 100
                processes = []
                file_lock = multiprocessing.Lock()
                hournodelist = list(graph_1hour.nodes())
                sublists = np.array_split(hournodelist[start:end], num_processes)
                with multiprocessing.Manager() as manager:
                    results = manager.list()

                    for i in range(num_processes):
                        sublist = sublists[i]
                        process = multiprocessing.Process(target=self.generate_ego_net_wrapper,
                                                          args=(self.graph, sublist, results, self.benford_G,self.Tgraph,file_lock))
                        processes.append(process)
                        process.start()

                    for process in processes:
                        process.join()

                    sorted_list = sorted(results, key=lambda x: x[0])
                    sorted_list_without_first_element = [item[1:] for item in sorted_list]
                graphs = sorted_list_without_first_element
                first_numbers = [item.pop(0) for item in graphs]
                end_time_ego = time.time()
                execution_time_ego = end_time_ego - start_time_ego
                print("k-ego net and chisquares: ", execution_time_ego, "秒")
                node_dict = {}
                start_time = time.time()
                subchi = first_numbers
                a=0
                for chi in subchi:
                    node_dict[start + a] = chi
                    a += 1

                graph_dict = {}

                for sublist, number in zip(graphs, subchi):
                    graph_dict[tuple(sublist)] = number

                node_sorted_keys = sorted(node_dict, key=node_dict.get, reverse=True)
                top_1000_node = node_sorted_keys[:k]
                sorted_dict = sorted(graph_dict,key=graph_dict.get , reverse=True)

                top_1000_graph = [list(key) for key in sorted_dict[:k]]
                graphemb = []
                for graph_id in top_1000_graph:
                    graphemb.append(self.graph.subgraph(graph_id))
                tmb_emb = generate_embedding(graphemb,self.model, self.Tgraph,device=self.args.device)
                for i in range(k):
                    all_emb[top_1000_node[i],:] = tmb_emb[i]

                print(
                    "No.{}-{} candidate AbnormalSubgraphs embedding finish".format(start, end))
                end_time = time.time()
                execution_time = end_time - start_time
                print("embedding time: ", execution_time, "秒")
        np.save(self.args.writer_dir + "/emb", all_emb)
        np.save(self.args.writer_dir + "/query", query_emb)
        return all_emb, query_emb


    def make_prediction(self,graph, seen_nodes,Tgraph):
        all_emb, query_emb = self.load_embedding()
        print(f"[Load Embedding], All shape {all_emb.shape}, Query shape {query_emb.shape}")

        pred_abnormalsubgraphs = []

        pred_size = 50
        single_pred_size = int(pred_size / query_emb.shape[0])
        print("single_pred_sized")
        print(single_pred_size)

        seeds = []

        for i in tqdm(range(query_emb.shape[0]), desc="Matching AbnormalSubgraphs"):
            q_emb = query_emb[i, :]

            distance = np.sqrt(np.sum(np.asarray(q_emb - all_emb) ** 2, axis=1))
            sort_dic = list(np.argsort(distance))

            if len(pred_abnormalsubgraphs) >= pred_size:
                break

            length = 0
            for node in sort_dic:
                if length >= single_pred_size:
                    break
                neighs = generate_ego_net(graph, node, self.benford_G,Tgraph,k=self.args.kego,max_size=50,
                                          choice="neighbors")

                if neighs not in pred_abnormalsubgraphs and len(pred_abnormalsubgraphs) < pred_size and node not in seen_nodes and node \
                        not in seeds:
                    seeds.append(node)
                    pred_abnormalsubgraphs.append(neighs)
                    length += 1
        lengths = np.array([len(pred_sub) for pred_sub in pred_abnormalsubgraphs])
        print(f"[Generate] Pred size {len(pred_abnormalsubgraphs)}, Avg Length {np.mean(lengths):.04f}")
        return pred_abnormalsubgraphs, all_emb
