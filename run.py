import json
from Discovery import AbnormalMatching
from Refinement import AbnormalRefine
import argparse
from datetime import datetime
import random
import numpy as np
import torch
from utils import load, feature_augmentation, split_abnormalsubgraphs,eval_scores_timeporal, eval_scores, load_benford,load_syn, loadTime
import os
import random, collections
import math
import time
import networkx as nx
from scipy import stats
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write2file(subs, filename):
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com]) for com in subs])
        fh.write(content)


def read4file(filename):
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')] for x in file.read().strip().split('\n')]
    return pred

def kl_divergence(p, q):
        return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if p[i] > 0)

def count_occ(l, dist_len, adjust_idx=0):
        result = [0 for i in range(dist_len)]
        for e in l:
            result[e - adjust_idx] += 1
        return result

def node_chisquare(edgelistl, node_num, dist, adjust_idx=1, directed='both', diff_func='chi'):
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
            count_dist = count_occ(node_dist, len(dist), adjust_idx)
            #        get the chi square statistic, the higher it is, more abnormal the node is
            if diff_func == 'chi':
                node_chis.append(stats.chisquare(count_dist, sum(count_dist) * np.array(dist))[0])
            elif diff_func == 'kl':
                node_chis.append(kl_divergence(np.array(count_dist) / sum(count_dist), np.array(dist)))
        return node_chis



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Abnormal Subgraph Encoder and Coarse Prototype Discovery related
    parser.add_argument("--conv_type", type=str, help="type of convolution", default="GCN")
    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=5)
    parser.add_argument("--kego", type=int, help="k-ego", default=5)
    parser.add_argument("--hidden_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--output_dim", type=int, help="training hidden size", default=64)
    parser.add_argument("--dropout", type=float, help="dropout rate", default=0.2)
    parser.add_argument("--margin", type=float, help="margin loss", default=0.4)
    parser.add_argument("--fine_ratio", dest="fine_ratio", type=float, help="fine-grained sampling ratio", default=0.0)

    # Train Abnormal Subgraph Encoder and Coarse Prototype Discovery
    parser.add_argument("--lr", dest="lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--device", dest="device", type=str, help="training device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, help="training batch size", default=32)
    # parser.add_argument("--batch_size", type=int, help="training batch size", default=16)
    parser.add_argument("--pairs_size", type=int, help="pairs size", default=10)
    # parser.add_argument("--pairs_size", type=int, help="pairs size", default=2)
    parser.add_argument("--seed", type=int, help="seed", default=0)

    parser.add_argument("--pred_size", type=int, help="pred size", default=50)
    parser.add_argument("--subm_path", type=str, help="suubmM path", default="")

    parser.add_argument("--dataset", type=str, help="dataset", default="eth-2018time7000")
    # parser.add_argument("--dataset", type=str, help="dataset", default="eth-2018jan")
    # parser.add_argument("--dataset", type=str, help="dataset", default="dblp-benford")

    # Train Anomalous Subgraph Refinement
    parser.add_argument("--agent_lr", type=float, help="submR learning rate", default=1e-3)
    parser.add_argument("--n_episode", type=int, help="number of episode", default=50)
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=100)
    parser.add_argument("--gamma", type=float, help="submR gamma", default=0.99)
    parser.add_argument("--max_step", type=int, help="", default=25)
    parser.add_argument("--time_intervial", type=int, help="", default=10)

    # Save log
    parser.add_argument("--writer_dir", type=str, help="Summary writer directory", default="")

    args = parser.parse_args()
    seed_all(args.seed)

    if not os.path.exists(f"ckpts/{args.dataset}"):
        os.mkdir(f"ckpts/{args.dataset}")
    args.writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.ab_size = 90 if args.dataset.startswith("lj") else 12
    print(args.writer_dir)

    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    nodes, edges, abnormalsubgraphs = load(args.dataset)
    # print(edges)
    edgelist , benford_G = load_benford()
    #load_benford use for eth-2019jan and eth-2018jan
    # edgelist, benford_G = load_syn()
    #load_syn use for dblp-benford dataset
    Tgraph = nx.Graph()
    tnodes,tedges = loadTime(args.dataset)
    for (u, v, weight) in tedges:
        # 检查图中是否已存在相同节点的边
        if Tgraph.has_edge(u, v):
            # 如果存在相同节点的边，则获取已存在的边的权重列表，并将新的权重添加到列表中
            current_weight = Tgraph[u][v]['weight']
            if isinstance(current_weight, list):
                current_weight.append(weight)
            else:
                # 如果当前权重不是列表，则将其转换为列表并添加新的权重
                Tgraph[u][v]['weight'] = [current_weight, weight]
        else:
            # 如果不存在相同节点的边，则添加新的边
            Tgraph.add_edge(u, v, weight=[weight])



    x = [e[2] for e in edgelist]
    x = collections.Counter(x)
    num_occ = np.array([x[k] for k in range(1, 10)]) / sum([x[k] for k in range(1, 10)])
    node_chisquares = node_chisquare(edgelist, benford_G.number_of_nodes(), num_occ, adjust_idx=1,
                                          diff_func='kl')


    graph, _ = feature_augmentation(nodes,edges)
    train_abnormalsubgraphs, val_abnormalsubgraphs, test_abnormalsubgraphs = split_abnormalsubgraphs(abnormalsubgraphs, 30, 10)
    # Training AbnormalMatching of Discovery
    abnormalM_obj = AbnormalMatching(args, graph, train_abnormalsubgraphs, val_abnormalsubgraphs, edges, edgelist, benford_G, node_chisquares,Tgraph)
    start_time = time.time()
    abnormalM_obj.train_epoch(1)
    end_time = time.time()
    execution_time = end_time - start_time
    seen_nodes = {node for com in train_abnormalsubgraphs + val_abnormalsubgraphs for node in com}
    pred_abnormalsubgraphs, feat_mat = abnormalM_obj.make_prediction(graph, seen_nodes,Tgraph)

    f, j, nmi = eval_scores_timeporal(graph, pred_abnormalsubgraphs, test_abnormalsubgraphs, benford_G, Tgraph,tmp_print=True)
    metrics_string = '_'.join([f'{x:0.4f}' for x in [f, j, nmi]])
    write2file(pred_abnormalsubgraphs, args.writer_dir + "/submM_" + metrics_string + '.txt')

    # Use F1-score as Reward function
    # Train Anomalous Subgraph Refinement
    cost_choice = "f1"
    abnormalR_obj = AbnormalRefine(args, graph, feat_mat, train_abnormalsubgraphs, val_abnormalsubgraphs, pred_abnormalsubgraphs, cost_choice, benford_G, node_chisquares,Tgraph)
    abnormalR_obj.train()
    refine_abnormalsubgraphs = abnormalR_obj.get_refine()
    f, j, nmi = eval_scores_timeporal(graph, refine_abnormalsubgraphs, test_abnormalsubgraphs, benford_G, Tgraph,tmp_print=True)
    metrics_string = '_'.join([f'{x:0.4f}' for x in [f, j, nmi]])
    write2file(refine_abnormalsubgraphs, args.writer_dir + f"/submR_{cost_choice}_" + metrics_string + '.txt')

    # Save setting
    with open(args.writer_dir + '/settings.json', 'w') as fh:
        arg_dict = vars(args)
        json.dump(arg_dict, fh, sort_keys=True, indent=4)

    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    # print('= ' * 20)
