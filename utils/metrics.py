from math import log
from typing import List, Union, Set
import numpy as np
import collections,math
import scipy.stats as stats
import networkx as nx

def compare_subgraph(pred_subgraphs: Union[List, Set],
                     true_subgraphs: Union[List, Set]) -> (float, float, float, float):
    """
    Compute the Precision, Recall, F1 and Jaccard similarity
    as the second argument is the ground truth AbnormalSubgraphs.
    """
    intersect = set(true_subgraphs) & set(pred_subgraphs)
    p = len(intersect) / len(pred_subgraphs)
    r = len(intersect) / len(true_subgraphs)
    f = 2 * p * r / (p + r + 1e-9)
    j = len(intersect) / (len(pred_subgraphs) + len(true_subgraphs) - len(intersect))
    return p, r, f, j

def get_start_digit(v):
    if v==0:
        return 0
    if v<0:
        v = -v
    while v<1:
        v = v*10
    return int(str(v)[:1])

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


def eval_scores(graph, pred_subgraphs, true_subgraphs, benford_G, tmp_print=False):
    # 4 columns for precision, recall, f1, jaccard
    pred_scores = np.zeros((len(pred_subgraphs), 4))
    truth_scores = np.zeros((len(true_subgraphs), 4))
    print("pred:")
    # print(pred_abnormalsubgraphs)
    x = 0
    density = []
    benford = []
    xs = [i for i in range(1, 10)]
    for i in range(9):
        benford.append(math.log10(1 + 1 / (i + 1)))
    idx = 0
    totalchi = []
    total_fei = []
    total_den = []
    for i in pred_subgraphs:
        gr = graph.subgraph(i)
        # print(gr)
        weights = 0
        for u, v, data in gr.edges(data=True):
            # 获取边的权重
            weight = data.get('weight')
            weights += weight
        nodes = len(gr.nodes())
        density.append((weights) / (nodes))
        x += 1

        interest = benford_G.subgraph(i)
        t = nx.get_edge_attributes(interest, 'weight')
        el = []
        interest = nx.convert_node_labels_to_integers(interest)
        for e in interest.edges():
            for w in interest[e[0]][e[1]]['weight']:
                el.append([e[0], e[1], get_start_digit(w)])

        node_chisquares1 = node_chisquare(el, interest.number_of_nodes(), benford, adjust_idx=1, diff_func='chi')
        avg_chi = []
        et = 0
        for i in range(interest.number_of_nodes()):
            tmp_deg = sum([len(interest[i][j]['weight']) for j in interest[i]])
            et += tmp_deg
            #         if tmp_deg<5:
            #             continue
            avg_chi.append(node_chisquares1[i] / tmp_deg)

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
        chi = 0
        for i in range(9):
            denominator = benford[i] * times
            if denominator != 0:
                chi += (obs[i] - denominator) ** 2 / denominator
            else:
                # 处理分母为零的情况
                chi += 0  # 或者其他适当的处理方式
        # chi = sum([(obs[i] - benford[i] * times) ** 2 / (benford[i] * times) for i in range(9)])
        lr = -2 * sum([obs[i] * math.log(benford[i] / (obs[i] / times)) for i in range(9) if obs[i] != 0])
        pval1 = 1 - stats.chi2.cdf(chi, 8)
        pval2 = 1 - stats.chi2.cdf(lr, 8)
        # print(idx,chi, pval1, lr, pval2, c3 / interest.number_of_nodes(), et / 2 / interest.number_of_nodes())
        print(idx, chi, c3 / interest.number_of_nodes(), et / 2 / interest.number_of_nodes())
        totalchi.append(chi)
        total_fei.append(c3/interest.number_of_nodes())
        total_den.append(et/2/interest.number_of_nodes())
        idx += 1
    print("acg:" + str(sum(density) / len(density)))
    print("avg chi:"+str(sum(totalchi) / len(totalchi)))
    print("avg chi:"+str(totalchi))
    print("avg fei:" + str(np.nanmean(total_fei)))
    print("avg fei:" + str(total_fei))
    print("avg den:" + str(sum(total_den) / len(total_den)))
    print("avg den:" + str(total_den))



    print("truth:")
    # print(true_subgraphs)

    for i, pred_comm in enumerate(pred_subgraphs):
        np.max([compare_subgraph(pred_comm, true_subgraphs[j])
                for j in range(len(true_subgraphs))], 0, out=pred_scores[i])

    for j, true_comm in enumerate(true_subgraphs):
        np.max([compare_subgraph(pred_subgraphs[i], true_comm)
                for i in range(len(pred_subgraphs))], 0, out=truth_scores[j])
    truth_scores[:, :2] = truth_scores[:, [1, 0]]


    # Avg F1 / Jaccard
    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.

    # detect percent
    comm_nodes = {node for com in true_subgraphs for node in com}
    pred_nodes = {node for com in pred_subgraphs for node in com}
    percent = len(list(comm_nodes & pred_nodes)) / len(comm_nodes)

    # NMI
    nmi_score = get_nmi_score(pred_subgraphs, true_subgraphs)

    print("--------------------------------------------")
    return round(mean_score_all[2], 4), round(mean_score_all[3], 4), round(nmi_score, 4)


def eval_scores_timeporal(graph, pred_subgraphs, true_subgraphs, benford_G, Tgraph, tmp_print=False):
    # 4 columns for precision, recall, f1, jaccard
    pred_scores = np.zeros((len(pred_subgraphs), 4))
    truth_scores = np.zeros((len(true_subgraphs), 4))
    print("pred:")
    x = 0
    temporal_density = []
    benford = []
    xs = [i for i in range(1, 10)]
    for i in range(9):
        benford.append(math.log10(1 + 1 / (i + 1)))
    idx = 0
    totalchi = []
    total_fei = []
    total_den = []
    TD = 0
    length_timeslice = 0
    for subgraph_nodes in pred_subgraphs:

        gr = graph.subgraph(subgraph_nodes)
        gr_time = Tgraph.subgraph(subgraph_nodes)
        weights = 0
        timestamp = []
        for u, v, data in gr.edges(data=True):
            # 获取边的权重
            weight = data.get('weight')
            weights += weight
        nodes = len(gr.nodes())
        for u,v,data in gr_time.edges(data=True):
            oneEdgeTimestamp = data.get('weight')
            for time in oneEdgeTimestamp:
                timestamp.append(time)


        # print("timestamp"+str(timestamp))
        # print("nodes:"+str(nodes))
        # print("weights"+str(weights))
        unique_timestamps = list(set(timestamp))
        time_slice_size = 4
        min_timestamp = unique_timestamps[0]
        time_slices = [(min_timestamp + i * time_slice_size) for i in range((unique_timestamps[-1] - min_timestamp) // time_slice_size + 1)]
        TD = (2 * weights) / (nodes * (nodes - 1) * len(unique_timestamps))
        print("time_slices: " + str(len(time_slices)))
        temporal_density.append(TD)
        length_timeslice = len(time_slices)
        if len(time_slices) == 0:
            length_timeslice = 1


        x += 1

        interest = benford_G.subgraph(subgraph_nodes)
        t = nx.get_edge_attributes(interest, 'weight')
        el = []
        interest = nx.convert_node_labels_to_integers(interest)
        for e in interest.edges():
            for w in interest[e[0]][e[1]]['weight']:
                el.append([e[0], e[1], get_start_digit(w)])

        node_chisquares1 = node_chisquare(el, interest.number_of_nodes(), benford, adjust_idx=1, diff_func='chi')
        avg_chi = []
        et = 0
        for i in range(interest.number_of_nodes()):
            tmp_deg = sum([len(interest[i][j]['weight']) for j in interest[i]])
            et += tmp_deg
            avg_chi.append(node_chisquares1[i] / tmp_deg)

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
        chi = 0
        for i in range(9):
            denominator = benford[i] * times
            if denominator != 0:
                chi += (obs[i] - denominator) ** 2 / denominator
            else:
                chi += 0
        lr = -2 * sum([obs[i] * math.log(benford[i] / (obs[i] / times)) for i in range(9) if obs[i] != 0])
        pval1 = 1 - stats.chi2.cdf(chi, 8)
        pval2 = 1 - stats.chi2.cdf(lr, 8)
        print(idx, chi, c3 / interest.number_of_nodes(), et / 2 / interest.number_of_nodes()/length_timeslice,TD)
        totalchi.append(chi)
        total_fei.append(c3 / interest.number_of_nodes())
        total_den.append(et / 2 / interest.number_of_nodes())
        idx += 1

    print("avg TD:" + str(sum(temporal_density) / len(temporal_density)))
    print("avg chi:" + str(sum(totalchi) / len(totalchi)))
    print("avg chi:" + str(totalchi))
    print("avg fei:" + str(np.nanmean(total_fei)))
    print("avg fei:" + str(total_fei))
    # print("avg den:" + str(sum(total_den) / len(total_den)))
    # print("avg den:" + str(total_den))

    print("truth:")

    for i, pred_comm in enumerate(pred_subgraphs):
        np.max([compare_subgraph(pred_comm, true_subgraphs[j])
                for j in range(len(true_subgraphs))], 0, out=pred_scores[i])

    for j, true_comm in enumerate(true_subgraphs):
        np.max([compare_subgraph(pred_subgraphs[i], true_comm)
                for i in range(len(pred_subgraphs))], 0, out=truth_scores[j])
    truth_scores[:, :2] = truth_scores[:, [1, 0]]

    # Avg F1 / Jaccard
    mean_score_all = (pred_scores.mean(0) + truth_scores.mean(0)) / 2.

    # detect percent
    comm_nodes = {node for com in true_subgraphs for node in com}
    pred_nodes = {node for com in pred_subgraphs for node in com}
    percent = len(list(comm_nodes & pred_nodes)) / len(comm_nodes)

    # NMI
    nmi_score = get_nmi_score(pred_subgraphs, true_subgraphs)

    print("--------------------------------------------")
    return round(mean_score_all[2], 4), round(mean_score_all[3], 4), round(nmi_score, 4)


def get_intersection(a, b, choice=None):
    return len(list(set(a) & set(b))) if not choice else list(set(a) & set(b))


def get_difference(a, b):
    intersection = get_intersection(a, b, choice="List")
    nodes = {x for x in a if x not in intersection}
    return len(list(nodes))


def get_nmi_score(pred, gt):
    def get_overlapping(pred_comms, ground_truth):
        """All nodes number"""
        nodes = {node for com in pred_comms + ground_truth for node in com}
        return len(nodes)

    def h(x):
        return -1 * x * (log(x) / log(2)) if x > 0 else 0

    def H_func(comm):
        p1 = len(comm) / overlapping_nodes
        p0 = 1 - p1
        return h(p0) + h(p1)

    def h_xi_joint_yj(xi, yj):
        p11 = get_intersection(xi, yj) / overlapping_nodes
        p10 = get_difference(xi, yj) / overlapping_nodes
        p01 = get_difference(yj, xi) / overlapping_nodes
        p00 = 1 - p11 - p10 - p01

        if h(p11) + h(p00) >= h(p01) + h(p10):
            return h(p11) + h(p10) + h(p01) + h(p00)
        return H_func(xi) + H_func(yj)

    def h_xi_given_yj(xi, yj):
        return h_xi_joint_yj(xi, yj) - H_func(yj)

    def H_XI_GIVEN_Y(xi, Y):
        res = h_xi_given_yj(xi, Y[0])
        for y in Y:
            res = min(res, h_xi_given_yj(xi, y))
        return res / H_func(xi)

    def H_X_GIVEN_Y(X, Y):
        res = 0
        # for idx in tqdm(range(len(X)), desc="ComputeNMI"):
        for idx in range(len(X)):
            res += H_XI_GIVEN_Y(X[idx], Y)
        return res / len(X)

    if len(pred) == 0 or len(gt) == 0:
        return 0

    overlapping_nodes = get_overlapping(pred, gt)
    return 1 - 0.5 * (H_X_GIVEN_Y(pred, gt) + H_X_GIVEN_Y(gt, pred))
