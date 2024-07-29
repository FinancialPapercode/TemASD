import random
from random import choice
import numpy as np
import sys
import scipy.stats as stats
import torch
import networkx as nx
import operator,collections,math
import datetime
from collections import deque
# from binary_heap import *
from collections import defaultdict

from deepsnap.graph import Graph as DSGraph, Graph
from deepsnap.batch import Batch
from torch_geometric.data import Data
import time


def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def find_weights(edge_list, other_variable):
    weights = []
    for edge in edge_list:
        for item in other_variable:
            if (edge[0] == item[0] and edge[1] == item[1]) or (edge[1] == item[0] and edge[0] == item[1]):
                weights.append(item[2])
                break
    return weights





def get_weights_from_edges(node_list, edges):
    node_set = set(node_list)
    weights = []
    edge_info = []

    for edge in edges:
        source_node, target_node, weight = edge[:3]

        if source_node in node_set and target_node in node_set:
            weights.append(weight)
            edge_info.append([source_node, target_node, weight])

    return weights, edge_info


def get_edges_from_nodes(neigh, edges):
    node_edges = {}
    for edge in edges:
        source_node, target_node, weight = edge[:3]
        node_pair = (source_node, target_node)
        if node_pair not in node_edges:
            node_edges[node_pair] = []
        node_edges[node_pair].append(weight)

    edges_info = []
    for node_pair in neigh:
        if node_pair in node_edges:
            edges_info.append(node_edges[node_pair])

    return edges_info



def has_edges_with_vertex(edge_info, vertex):
    for edge in edge_info:
        if vertex in edge[:2]:
            return True
    return False



def process_edges(edge_info, num, subgraph_vertices):
    flag = num
    i=0

    while flag > 0:
        i += 1
        if i > 700:
            break;

        w = []
        for edge in edge_info:
            w.append(edge[2])

        selected_edge = random.choices(edge_info, weights=w)[0]



        temp_edge_info = edge_info[:]
        updated_subgraph_vertices = subgraph_vertices[:]

        node1, node2, weight = selected_edge

        if weight >= 2:
            selected_edge[2] -= 1
            weight -= 1
            index = temp_edge_info.index(selected_edge)
            temp_edge_info[index][2] = weight
            flag -= 1

        else:
            if (node1 in updated_subgraph_vertices) and (node2 in updated_subgraph_vertices) and node1 != node2:
                temp_edge_info.remove(selected_edge)
                if not has_edges_with_vertex(temp_edge_info,node1):
                    updated_subgraph_vertices.remove(node1)
                if not has_edges_with_vertex(temp_edge_info,node2):
                    updated_subgraph_vertices.remove(node2)

                graph = nx.Graph()
                graph.add_nodes_from(updated_subgraph_vertices)
                graph.add_weighted_edges_from(temp_edge_info)
                if graph.number_of_nodes() > 0:
                    is_connected = nx.is_connected(graph)
                else:
                    is_connected = False


                if not is_connected:
                    temp_edge_info.append(selected_edge)
                    if node1 not in updated_subgraph_vertices:
                        updated_subgraph_vertices.append(node1)
                    if node2 not in updated_subgraph_vertices:
                        updated_subgraph_vertices.append(node2)
                else:
                    flag -= 1

    return temp_edge_info, updated_subgraph_vertices


def sample_neigh(graphs, size, edges, graph_weight):
    """Sampling function during training"""
    ps = np.array([len(g) for g in graphs], dtype=np.float64)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

    while True:
        idx = dist.rvs()
        graph = graphs[idx]
        edge_info = []
        we = graph_weight
        neigh = []
        for edge in graph.edges(data=True):
            edge_info.append([edge[0], edge[1], edge[2]['weight']])
        total = list(set(graph.nodes))
        for node in total:
            neigh.append(node)

        num = abs(size - graph_weight)
        if num != 0 and num < we:
            updated_edge_info, neigh = process_edges(edge_info, num, neigh)
        else:
            updated_edge_info = edge_info
        return graph, neigh, updated_edge_info

def merge_adjacent_duplicates(edges):
    merged_edges = []
    prev_edge = None
    count = 0

    for edge in edges:
        if edge == prev_edge:
            count += 1
        else:
            if prev_edge is not None:
                merged_edges.append((prev_edge, count))
            prev_edge = edge
            count = 1

    if prev_edge is not None:
        merged_edges.append((prev_edge, count))

    return merged_edges

def prefix_sum(merged_edges):
    n = len(merged_edges)
    prefix_counts = [0] * (n + 1)

    # 计算前缀和
    for i in range(1, n + 1):
        prefix_counts[i] = prefix_counts[i - 1] + merged_edges[i - 1][1]

    return prefix_counts

def max_edges_in_slice(prefix_counts, k):
    max_edges = 0
    max_start = 0

    # 计算每个时间切片中包含的边数量
    for i in range(k, len(prefix_counts)):
        edges_in_slice = prefix_counts[i] - prefix_counts[i - k]
        if edges_in_slice > max_edges:
            max_edges = edges_in_slice
            max_start = i - k

    return max_start, max_edges


def reconstruct_subgraph(graph, start_index, slice_length):
    subgraph_edges = []

    for u, v, data in graph.edges(data=True):
        if isinstance(data['weight'], list):
            for edge in data['weight']:
                if start_index <= edge < start_index + slice_length:
                    subgraph_edges.append((u, v, edge))
        else:
            if start_index <= data['weight'] < start_index + slice_length:
                subgraph_edges.append((u, v, data['weight']))

    subgraph = nx.DiGraph()
    for (u, v, weight) in subgraph_edges:
        if subgraph.has_edge(u, v):
            current_weight = subgraph[u][v]['weight']
            if isinstance(current_weight, list):
                current_weight.append(weight)
            else:
                subgraph[u][v]['weight'] = [current_weight, weight]
        else:
            subgraph.add_edge(u, v, weight=[weight])

    return subgraph


def max_density_subgraph(edges, interval):
    edges.sort(key=lambda x: x[2])
    n = len(edges)
    max_density = 0
    max_density_subgraph = None
    start_time = edges[0][2]
    end_time = edges[-1][2]

    nodes_set = set()
    edges_count = 0
    left = 0

    for right in range(n):
        while edges[right][2] - edges[left][2] >= interval:
            nodes_set.discard(edges[left][0])
            nodes_set.discard(edges[left][1])
            edges_count -= 1
            left += 1
        nodes_set.update(edges[right][:2])
        edges_count += 1
        if len(nodes_set) == 0:
            continue
        density = edges_count / len(nodes_set)
        if density > max_density:
            max_density = density
            max_density_subgraph = (edges[left][2], edges[right][2])

    return max_density_subgraph
def compare_node_ids(before_graph, after_graph):
    before_node_ids = set(before_graph)
    after_node_ids = set(after_graph)

    added_nodes = after_node_ids - before_node_ids

    removed_nodes = before_node_ids - after_node_ids

    return added_nodes, removed_nodes
def generate_ego_net(graph, start_node, benford_G,Tgraph,k=1, max_size=15, choice="subgraph"):
    """Generate **k** ego-net"""
    q = [start_node]
    visited = [start_node]

    iteration = 0
    while True:
        if iteration >= k:
            break
        length = len(q)
        if length == 0 or len(visited) >= max_size:
            break

        for i in range(length):
            # Queue pop
            u = q[0]
            q = q[1:]

            for v in list(graph.neighbors(u)):
                if v not in visited:
                    q.append(v)
                    visited.append(v)
                if len(visited) >= max_size:
                    break
            if len(visited) >= max_size:
                break
        iteration += 1
    visited = sorted(visited)
    # print("visited:"+str(visited))

    visited_graph = Tgraph.subgraph(visited)

    visited_graph_alledge = []
    after_graph = []
    for u, v, data in visited_graph.edges(data=True):
        weights = data['weight']
        if not isinstance(weights, list):
            weights = [weights]

        for weight in weights:
            one_edge = [u, v, weight]
            visited_graph_alledge.append(one_edge)

    G_qtcs = qtcsGraph(visited_graph_alledge)
    alpha=0.2
    peel_visited = []
    s = int(start_node)
    if s in list(G_qtcs.tadj_list):
        seed = s
    else:
        seed = int(random.choice(list(G_qtcs.tadj_list)))

    result, time_tppr, t_egr = G_qtcs.EGR(alpha, seed)
    peel_visited = list(result)
    if peel_visited == []:
        peel_visited = list(visited_graph.nodes)
    if choice =="neighbors":
        return peel_visited
    elif choice =="multi":
        multi = [start_node]
        benford = []
        for i in range(9):
            benford.append(math.log10(1 + 1 / (i + 1)))
        sub = benford_G.subgraph(peel_visited)
        t = nx.get_edge_attributes(sub, 'weight')
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
        chi = 0
        denominators = np.array(benford) * times
        non_zero_indices = np.nonzero(denominators)

        for i in range(9):
                denominator = benford[i] * times
                if denominator != 0:
                    chi += (obs[i] - denominator) ** 2 / denominator
                else:

                    chi += 0
        multi.append(chi)
        multi = multi + peel_visited
        return multi
    else:
        return graph.subgraph(peel_visited)

def generate_outer_boundary_3(graph, com_nodes, max_size=20):
    outer_nodes = []

    for node in com_nodes:
        outer_nodes += list(graph.neighbors(node))
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    outer_nodes = sorted(outer_nodes)

    outer_nodes_2 = {}
    for node in com_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor not in com_nodes:
                weight = graph[node][neighbor]['weight']
                outer_nodes_2[neighbor] = weight
    sorted_keys = sorted(outer_nodes_2.keys(), key=lambda x: outer_nodes_2[x], reverse=True)

    if len(outer_nodes)<=max_size:
        new_array = create_new_array(outer_nodes,sorted_keys,len(outer_nodes))
    else:
        new_array = create_new_array(outer_nodes,sorted_keys,max_size)

    return new_array

def generate_outer_boundary_with_dynamic_threshold(graph, com_nodes, Tgraph, max_size=20):
    outer_nodes = []
    latest_timestamps = [Tgraph[node][neighbor]['weight'][-1] for node in com_nodes for neighbor in graph.neighbors(node)]
    threshold = sum(latest_timestamps) / len(latest_timestamps)

    for node in com_nodes:
        outer_nodes += [neighbor for neighbor in graph.neighbors(node) if Tgraph[node][neighbor]['weight'][-1] > threshold]
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    outer_nodes = sorted(outer_nodes)

    outer_nodes_weights = {}
    for node in com_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor not in com_nodes and Tgraph[node][neighbor]['weight'][-1] > threshold:
                weight = graph[node][neighbor]['weight']
                outer_nodes_weights[neighbor] = weight

    sorted_keys = sorted(outer_nodes_weights.keys(), key=lambda x: outer_nodes_weights[x], reverse=True)

    if len(outer_nodes) <= max_size:
        new_array = create_new_array(outer_nodes, sorted_keys, len(outer_nodes))
    else:
        new_array = create_new_array(outer_nodes, sorted_keys, max_size)

    return new_array

def create_new_array(sorted_keys, outer_nodes, max_size):
    if len(sorted_keys) >= max_size and len(outer_nodes) >= max_size:
        selected_sorted_keys = sorted_keys[:10]
        selected_outer_nodes = outer_nodes[:10]
    else:
        selected_sorted_keys = sorted_keys[:len(sorted_keys) // 2]
        selected_outer_nodes = outer_nodes[:len(outer_nodes) // 2]

    new_array = selected_sorted_keys + selected_outer_nodes
    unique_elements = list(set(new_array))
    return unique_elements[:max_size]



def batch2graphs(graphs, TGraph,device=None):
    """Transform `List[nx.TGraph]` into `DeepSnap.Batch` object"""
    graph_data = [DSGraph(g) for g in graphs]
    for graph, graph_edge in zip(graph_data, graphs):
        weights = []
        edge_timestamps = []
        for u, v, data in graph_edge.edges(data=True):
            weight = data.get('weight')
            weights.append(weight)
        timegraph = TGraph.subgraph(list(graph_edge.nodes()))
        for t1,t2,data in timegraph.edges(data=True):
            onetime = data.get('weight')
            edge_timestamps.append(onetime)
        graph.weight = weights
        graph.edge_timestamps = edge_timestamps

    batch = Batch.from_data_list(graph_data)
    batch = batch.to(get_device(device=device))

    return batch


def batch2graphstimeslice(graphs, TGraph, time_interval, device=None):
    """Transform `List[nx.TGraph]` into `DeepSnap.Batch` object with time slicing"""
    graph_data = []

    # Step 1: Slice the input graphs based on time
    for graph in graphs:

        sliced_graphs = slice_graph_by_time(graph, TGraph, time_interval)
        if sliced_graphs == []:
            weights = []
            new_dsg = DSGraph(graph)
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight')
                weights.append(weight)
            new_dsg.weight = weights
            graph_data.append(new_dsg)
        else:
            for sliced_graph in sliced_graphs:
                # print(sliced_graph)
                weights = []

                if len(sliced_graph.nodes()) > 0:  # 添加条件判断
                    new_dsg = DSGraph(sliced_graph)
                    for u, v, data in sliced_graph.edges(data=True):
                        weight = data.get('weight')
                        weights.append(weight)
                    new_dsg.weight = weights
                    graph_data.append(new_dsg)


    batch = Batch.from_data_list(graph_data)
    batch = batch.to(get_device(device=device))

    return batch


def slice_graph_by_time(graph,TGraph, time_interval):
    sliced_graphs = []
    edge_timestamps = []
    timegraph = TGraph.subgraph(list(graph.nodes()))
    for t1, t2, data in timegraph.edges(data=True):
        onetime = data.get('weight')
        edge_timestamps.append(onetime)

    min_timestamp = min(edge_timestamps[0])
    max_timestamp = max(edge_timestamps[0])
    intervals = range(min_timestamp, max_timestamp, time_interval)
    if min_timestamp+time_interval >=max_timestamp:
        return [graph]
    else:
        for start_time, end_time in zip(intervals, intervals[1:]):
            timenode = find_nodes_in_time_slice(timegraph, start_time, end_time)
            timesubgraph = graph.subgraph(timenode)
            if timesubgraph.nodes() != [] and timesubgraph.edges() != []:
                sliced_graphs.append(graph.subgraph(timenode))
        return sliced_graphs


def find_nodes_in_time_slice(timegraph, start_time, end_time):
    nodes_in_time_slice = set()

    for u, v, data in timegraph.edges(data=True):
        edge_timestamps = data.get('weight')
        if edge_timestamps:
            for timestamp in edge_timestamps:
                if start_time <= timestamp < end_time:
                    nodes_in_time_slice.add(u)
                    nodes_in_time_slice.add(v)

    return list(nodes_in_time_slice)

def batch2graphstimeslice_nsubgraph(graphs, TGraph, n, device=None):
    """Transform `List[nx.TGraph]` into `DeepSnap.Batch` object with n time slices"""
    graph_data = []

    for graph in graphs:
        sliced_graphs = slice_graph_by_time_nsubgraph(graph, TGraph, n)
        if sliced_graphs == []:
            weights = []
            new_dsg = DSGraph(graph)
            for u, v, data in graph.edges(data=True):
                weight = data.get('weight')
                weights.append(weight)
            new_dsg.weight = weights
            graph_data.append(new_dsg)
        else:
            for sliced_graph in sliced_graphs:
                weights = []
                if len(sliced_graph.nodes()) > 0:
                    new_dsg = DSGraph(sliced_graph)
                    for u, v, data in sliced_graph.edges(data=True):
                        weight = data.get('weight')
                        weights.append(weight)
                    new_dsg.weight = weights
                    graph_data.append(new_dsg)

    batch = Batch.from_data_list(graph_data)
    batch = batch.to(get_device(device=device))

    return batch

def slice_graph_by_time_nsubgraph(graph, TGraph, n):
    sliced_graphs = []
    edge_timestamps = []
    timegraph = TGraph.subgraph(list(graph.nodes()))
    for t1, t2, data in timegraph.edges(data=True):
        onetime = data.get('weight')
        edge_timestamps.append(onetime)

    min_timestamp = min(edge_timestamps[0])
    max_timestamp = max(edge_timestamps[0])

    intervals = np.linspace(min_timestamp, max_timestamp, n+1)

    for i in range(n):
        start_time = intervals[i]
        end_time = intervals[i+1]

        timenode = find_nodes_in_time_slice(timegraph, start_time, end_time)
        timesubgraph = graph.subgraph(timenode)
        if list(timesubgraph.nodes()) != [] and list(timesubgraph.edges()) != []:
            sliced_graphs.append(graph.subgraph(timenode))

    while len(sliced_graphs) < n:
        sliced_graphs.append(graph.copy())

    return sliced_graphs


def generate_embedding(batch, model, TGraph,device=None):
    batches = batch2graphs(batch,TGraph, device=device)
    pred = model.encoder(batches)
    pred = pred.cpu().detach().numpy()
    return pred


def split_abnormalsubgraphs(subgraphs, n_train, n_val=0):
    print(f"Split abnormalsubgraphs, # Train {n_train}, # Val {n_val}, # Test {len(subgraphs) - n_train - n_val}")
    random.shuffle(subgraphs)
    return subgraphs[:n_train], subgraphs[n_train:n_train + n_val], subgraphs[n_train + n_val:]






class TGraph:

    def __init__(self, t_edges):
        self.edge = t_edges
        self.sadj_list, self.tadj_list, self.tmin, self.tmax = self.TemporalGraph()
        # print("number of nodes: " + str(len(self.sadj_list)))
        number1 = 0
        number2 = 0
        for u in self.sadj_list:
            number1 += len(self.sadj_list[u])
            for t in self.tadj_list[u]:
                number2 += len(self.tadj_list[u][t])

    def TemporalGraph(self):
        tmin, tmax, sadj_list, tadj_list = 1000000000000, -1, {}, {}
        for e in self.edge:
                from_id, to_id, time_id = int(e[0]), int(e[1]), int(e[2])
                if from_id == to_id:
                    continue
                if time_id > tmax:
                    tmax = time_id
                if time_id < tmin:
                    tmin = time_id
                for (f_id, t_id) in [(from_id, to_id), (to_id, from_id)]:  # 有向变无向.
                    if f_id in sadj_list:
                        if t_id not in sadj_list[f_id]:
                            sadj_list[f_id].append(t_id)
                    else:
                        sadj_list[f_id] = [t_id]
                    if f_id in tadj_list:
                        if time_id in tadj_list[f_id]:
                            if t_id not in tadj_list[f_id][time_id]:
                                tadj_list[f_id][time_id].append(t_id)
                        else:
                            tadj_list[f_id][time_id] = [t_id]  # 采用list较set内存消耗小
                    else:
                        tadj_list[f_id] = {}
                        tadj_list[f_id][time_id] = [t_id]
        return (sadj_list, tadj_list, tmin, tmax)

    def kcore(self, adj):
        q = []
        D = []
        deg = {}
        for node_id in adj:
            deg[node_id] = len(adj[node_id])
            if deg[node_id] < self.k:
                q.append(node_id)
        while (len(q) > 0):
            v = q.pop()
            D.append(v)
            for w in adj[v]:
                if deg[w] >= self.k:
                    deg[w] = deg[w] - 1
                    if deg[w] < self.k:
                        q.append(w)
        Vc = adj.keys() - set(D)
        return Vc

    def intersection_graph(self, tadj_list, interval, nodes):
        intersection_adj = {}
        interval_iterator = range(interval[0] + 1, interval[1] + 1)
        for node in nodes:
            if interval[0] in tadj_list[node]:
                intersection_adj[node] = set(tadj_list[node][interval[0]]) & nodes
            else:
                continue
            for timestamp in interval_iterator:
                if timestamp in tadj_list[node]:
                    intersection_adj[node] &= set(tadj_list[node][timestamp])
                else:
                    del intersection_adj[node]
                    break
        Vc = self.kcore(intersection_adj)
        return [interval, Vc]

    def Naive(self, k, sigma):
        self.k, self.sigma = k, sigma
        # print(self.edge + "naive")
        starttime = datetime.datetime.now()
        LC = []
        for time_gap in range(self.sigma, self.tmax - self.tmin + 2):
            for timestamp in range(self.tmin, self.tmax + 1 - time_gap + 1):
                temp = self.intersection_graph(self.tadj_list, (timestamp, timestamp + time_gap - 1),
                                               set(self.sadj_list.keys()))
                if len(temp[1]) > 0:
                    LC.append(temp)
        for result in LC[:]:  # maximal check
            for candi in LC[:]:
                if candi != result and set(result[1]).issubset(set(candi[1])) and candi[0][0] <= result[0][0] and \
                        candi[0][1] >= result[0][1]:
                    LC.remove(result)
                    break
        endtime = datetime.datetime.now()
        # print('time (s): ' + str(endtime - starttime))
        # print(len(LC))

    def Enum(self, k, sigma):
        self.k, self.sigma = k, sigma
        Vc = self.kcore(self.sadj_list)
        intersection_graph_count = [0]
        LC, Q = [], deque()
        for timestamp in range(self.tmin, self.tmax + 1 - self.sigma + 1):
            temp = self.intersection_graph(self.tadj_list, (timestamp, timestamp + self.sigma - 1), Vc)
            intersection_graph_count[0] = intersection_graph_count[0] + 1
            if len(temp[1]) > 0:
                Q.append(temp)
        while Q:
            l = len(Q)
            if l == 1:
                r = Q.popleft()
                LC.append(r)
                break
            R1 = Q.popleft()
            LC.append(R1)
            for _ in range(l - 1):
                R2 = Q.popleft()
                LC.append(R2)
                if R1[0][1] >= R2[0][0]:
                    nodelist = R1[1] & R2[1]
                    temp = self.intersection_graph(self.tadj_list, (R1[0][0], R2[0][1]), nodelist)
                    intersection_graph_count[0] = intersection_graph_count[0] + 1
                    if len(temp[1]) > 0:
                        Q.append(temp)
                R1 = R2
        candi_R = LC[:]
        for result in LC[:]:  # maximal check
            for candi in LC:
                if candi != result and set(result[1]).issubset(set(candi[1])) and candi[0][0] <= result[0][0] and \
                        candi[0][1] >= result[0][1]:
                    LC.remove(result)
                    break
        # print(len(LC))
        return LC, intersection_graph_count

    def GreLC(self, k, sigma, r):
        self.k, self.sigma, self.r = k, sigma, r
        # print(self.edge + "grelc")
        # print("efefffffffffffffffffffff")
        starttime = datetime.datetime.now()
        LC, intersection_graph_count = self.Enum(k, sigma)
        R = []
        if len(LC) <= self.r:
            R = LC
        else:
            max = -1
            for result in LC:
                if (result[0][1] - result[0][0] + 1) * len(result[1]) > max:
                    temp = result
                    max = (result[0][1] - result[0][0] + 1) * len(result[1])
            R.append(temp)
            LC.remove(temp)
            for _ in range(self.r - 1):
                max = -1
                b = len(self.coverage_measure(R))
                for result in LC:
                    R.append([result[0], result[1]])
                    a = len(self.coverage_measure(R))
                    R.remove([result[0], result[1]])
                    size = a - b
                    if size > max:
                        temp = result
                        max = size
                R.append(temp)
                LC.remove(temp)
        coverage = len(self.coverage_measure(R))
        endtime = datetime.datetime.now()
        # print("R: " + str(len(R)))
        # print(R)
        # print("intersection_graph_count" + str(intersection_graph_count))
        # print('time (s): ' + str(endtime - starttime) + "coverage" + str(coverage))

    def coverage_measure(self, can_R):
        coverage_set = set()
        for [I, V] in can_R:
            for t in range(I[0], I[1] + 1):
                for v in V:
                    coverage_set.add((v, t))
        return coverage_set

    def Pruning(self):
        s = datetime.datetime.now()
        Vc = self.kcore(self.sadj_list)
        D, Q, LI, d = set(), [], {}, {}
        for v in Vc:
            count = 0
            d[v] = {}
            for t in range(self.tmin, self.tmax + 1):
                if t in self.tadj_list[v]:
                    d[v][t] = len(Vc & set(self.tadj_list[v][t]))
                    if d[v][t] >= self.k:
                        count = count + 1
                    else:
                        Q.append((v, t))
                        d[v][t] = 0
                if t not in self.tadj_list[v] or d[v][t] < self.k:
                    if count < self.sigma:
                        for t1 in range(t - count, t):
                            Q.append((v, t1))
                            d[v][t1] = 0
                    else:
                        if v in LI:
                            LI[v].add((t - count, t - 1))
                        else:
                            LI[v] = {(t - count, t - 1)}
                    count = 0
                else:
                    if t == self.tmax:
                        if count < self.sigma:
                            for t1 in range(t - count + 1, t + 1):
                                Q.append((v, t1))
                                d[v][t1] = 0
                        else:
                            if v in LI:
                                LI[v].add((t - count + 1, t))
                            else:
                                LI[v] = {(t - count + 1, t)}
        while len(Q) > 0:
            (v, t) = Q.pop()
            D.add((v, t))
            for u in Vc & set(self.tadj_list[v][t]):
                if d[u][t] >= self.k:
                    d[u][t] = d[u][t] - 1
                    if d[u][t] < self.k:
                        Q.append((u, t))
                        d[u][t] = 0
                        self.UpdateLasting(LI, u, t, Q, d)
        tmin, tmax, tadj_list = 1000000000000, -1, {}
        for e in self.edge:
                from_id, to_id, time_id = int(e[0]), int(e[1]), int(e[2])
                if from_id == to_id:
                    continue
                if from_id in Vc and to_id in Vc:
                    if (from_id, time_id) not in D and (to_id, time_id) not in D:
                        if time_id > tmax:
                            tmax = time_id
                        if time_id < tmin:
                            tmin = time_id
                        for (f_id, t_id) in [(from_id, to_id), (to_id, from_id)]:
                            if f_id in tadj_list:
                                if time_id in tadj_list[f_id]:
                                    if t_id not in tadj_list[f_id][time_id]:
                                        tadj_list[f_id][time_id].append(t_id)
                                else:
                                    tadj_list[f_id][time_id] = [t_id]
                            else:
                                tadj_list[f_id] = {}
                                tadj_list[f_id][time_id] = [t_id]
        return [tadj_list, tmin, tmax, LI]

    def UpdateLasting(self, LI, u, t, Q, d):
        for (s, e) in LI[u]:
            if s <= t <= e:
                LI[u].remove((s, e))
                if t - s >= self.sigma:
                    LI[u].add((s, t - 1))
                else:
                    for t1 in range(s, t):
                        Q.append((u, t1))
                        d[u][t1] = 0
                if e - t >= self.sigma:
                    LI[u].add((t + 1, e))
                else:
                    for t2 in range(t + 1, e + 1):
                        Q.append((u, t2))
                        d[u][t2] = 0
                return

    def TopLC(self, k, sigma, r):
        self.k, self.sigma, self.r = k, sigma, r
        tadj_list1, tmin1, tmax1, LI = self.Pruning()
        # print(str(self.edge) + "toplc")
        starttime = datetime.datetime.now()
        self.R, self.A, self.B, self.flag = [], {}, {}, []
        search_count, intersection_graph_count = [0], [0]
        for timestamp in range(tmin1, tmax1 + 1 - self.sigma + 1):
            interval = (timestamp, timestamp + self.sigma - 1)
            P = set()
            for v in tadj_list1:
                P.add(v)
            self.search(interval, LI, P, tadj_list1, tmax1, search_count, intersection_graph_count)
        endtime = datetime.datetime.now()
        return self.R

    def search(self, interval, LI, S, tadj_list1, tmax1, search_count, intersection_graph_count):
        search_count[0] = search_count[0] + 1
        ts, te = interval[0], interval[1]
        D = set()
        for u in S:
            index = 0
            for (t1, t2) in LI[u]:
                if t1 <= ts and te <= t2:
                    break
                else:
                    index = index + 1
            if index == len(LI[u]):
                D.add(u)
        S.difference_update(D)
        if len(self.R) == self.r and len(S) * (tmax1 - ts + 1) < min(self.B.keys()) + len(self.A) / self.r:
            return
        temp = self.intersection_graph(tadj_list1, interval, S)
        intersection_graph_count[0] = intersection_graph_count[0] + 1
        if len(temp[1]) == 0:
            return
        if len(self.flag) == 0:
            self.flag.append(temp)
        else:
            H = self.flag.pop()
            self.flag.append(temp)
            if H[0][0] == ts and H[1] != temp[1]:
                for result in self.R:
                    if set(H[1]).issubset(set(result[1])) and result[0][0] <= H[0][0] and result[0][1] >= H[0][1]:
                        return
                self.Update(H)
            if H[0][0] != ts:
                self.Update(H)
        if te == tmax1:
            return
        else:
            self.search((ts, te + 1), LI, S, tadj_list1, tmax1, search_count, intersection_graph_count)

    def Update(self, temp):
        if len(self.R) < self.r:
            self.add(temp)
        else:
            if self.pcov(temp) >= min(self.B.keys()) + len(self.A) / self.r:
                self.delete()
                self.add(temp)

    def pcov(self, temp):
        count = 0
        R_min = self.B[min(self.B.keys())][0]
        for v in temp[1]:
            for t in range(temp[0][0], temp[0][1] + 1):
                if (v, t) not in self.A:
                    count = count + 1
                    continue
                if v in R_min[1] and R_min[0][0] <= t <= R_min[0][1] and len(self.A[(v, t)]) == 1:
                    count = count + 1
        return count

    def add(self, temp):
        self.R.append(temp)
        delta = 0
        for v in temp[1]:
            for t in range(temp[0][0], temp[0][1] + 1):
                if (v, t) not in self.A:
                    self.A[(v, t)] = [temp]
                    delta = delta + 1
                    continue
                if len(self.A[(v, t)]) == 1:
                    R = self.A[(v, t)][0]
                    for i in self.B:
                        if R in self.B[i]:
                            self.B[i].remove(R)
                            if len(self.B[i]) == 0:
                                del self.B[i]
                            if i - 1 in self.B:
                                self.B[i - 1].append(R)
                            else:
                                self.B[i - 1] = [R]
                            break
                self.A[(v, t)].append(temp)
        if delta not in self.B:
            self.B[delta] = [temp]
        else:
            self.B[delta].append(temp)

    def delete(self):
        min_pcov = min(self.B.keys())
        R_min = self.B[min_pcov].pop()
        self.R.remove(R_min)
        if len(self.B[min_pcov]) == 0:
            del self.B[min_pcov]
        for v in R_min[1]:
            for t in range(R_min[0][0], R_min[0][1] + 1):
                self.A[(v, t)].remove(R_min)
                if len(self.A[(v, t)]) == 1:
                    R = self.A[(v, t)][0]
                    for i in self.B:
                        if R in self.B[i]:
                            self.B[i].remove(R)
                            if len(self.B[i]) == 0:
                                del self.B[i]
                            if i + 1 in self.B:
                                self.B[i + 1].append(R)
                            else:
                                self.B[i + 1] = [R]
                            break
                    continue
                if len(self.A[(v, t)]) == 0:
                    del self.A[(v, t)]

    def TopLC_B(self, k, sigma, r):
        self.k, self.sigma, self.r = k, sigma, r
        print(self.edge + "toplc_b")
        tadj_list1, tmin1, tmax1, LI = self.Pruning()
        starttime = datetime.datetime.now()
        self.R, self.flag = [], []
        search_count, intersection_graph_count = [0], [0]
        for timestamp in range(tmin1, tmax1 + 1 - self.sigma + 1):
            interval = (timestamp, timestamp + self.sigma - 1)
            P = set()
            for v in tadj_list1:
                P.add(v)
            self.search_b(interval, LI, P, tadj_list1, tmax1, search_count, intersection_graph_count)
        endtime = datetime.datetime.now()
        print("search_count:" + str(search_count))
        print("intersection_graph_count" + str(intersection_graph_count))
        print("R: " + str(len(self.R)))
        print('time (s): ' + str(endtime - starttime) + "coverage" + str(len(self.coverage_measure(self.R))))

    def search_b(self, interval, LI, S, tadj_list1, tmax1, search_count, intersection_graph_count):
        search_count[0] = search_count[0] + 1
        ts, te = interval[0], interval[1]
        D = set()
        for u in S:
            index = 0
            for (t1, t2) in LI[u]:
                if t1 <= ts and te <= t2:
                    break
                else:
                    index = index + 1
            if index == len(LI[u]):
                D.add(u)
        S.difference_update(D)
        if te == tmax1:
            self.flag = []
            return
        if len(self.R) == self.r:
            cov_R = len(self.coverage_measure(self.R))
            min_size = 10000000000
            for result in self.R:
                tem = self.R[:]
                tem.remove(result)
                size = len(self.coverage_measure([result]) - self.coverage_measure(tem))
                if size < min_size:
                    min_size = size
            if len(S) * (tmax1 - ts + 1) < min_size + cov_R / self.r:
                self.flag = []
                return
        temp = self.intersection_graph(tadj_list1, interval, S)
        intersection_graph_count[0] = intersection_graph_count[0] + 1
        if len(temp[1]) == 0:
            self.flag = []
            return
        if self.flag == []:
            self.flag.append(temp)
        else:
            H = self.flag.pop()
            self.flag.append(temp)
            if H[1] != temp[1]:
                for result0 in self.R:
                    if set(H[1]).issubset(set(result0[1])) and result0[0][0] <= H[0][0] and result0[0][1] >= H[0][1]:
                        return
                self.Update_b(H)
        self.search_b((ts, te + 1), LI, S, tadj_list1, tmax1, search_count, intersection_graph_count)

    def Update_b(self, temp):
        if len(self.R) < self.r:
            self.R.append(temp)
        else:
            cov_R1 = len(self.coverage_measure(self.R))
            min_size1 = 10000000000
            for result1 in self.R:
                tem1 = self.R[:]
                tem1.remove(result1)
                size1 = len(self.coverage_measure([result1]) - self.coverage_measure(tem1))
                if size1 < min_size1:
                    min_size1 = size1
                    min_r = result1
            self.R.remove(min_r)
            self.R.append(temp)
            if len(self.coverage_measure(self.R)) < (1 + 1 / self.r) * cov_R1:
                self.R.remove(temp)
                self.R.append(min_r)

    def metric(self, R):
        AB, ATC = 0, 0
        for res in R:
            temp = 0
            for t in range(res[0][0], res[0][1] + 1):
                for v in res[1]:
                    if t in self.tadj_list[v]:
                        temp = temp + len(set(self.tadj_list[v][t]) & res[1])
            AB = AB + temp / (len(res[1]) * (len(res[1]) - 1) * (res[0][1] - res[0][0] + 1))
        AB = AB / len(R)
        for res in R:
            cut, vol, vol_bar = 0, 0, 0
            for t in range(res[0][0], res[0][1] + 1):
                for u in res[1]:
                    for v in set(self.sadj_list.keys()) - set(res[1]):
                        if v in self.tadj_list[u][t]:
                            cut = cut + 1
                    vol = vol + len(self.tadj_list[u][t])
                for w in set(self.sadj_list.keys()) - set(res[1]):
                    if t in self.tadj_list[w]:
                        vol_bar = vol_bar + len(self.tadj_list[w][t])
            if vol > vol_bar:
                vol = vol_bar
            ATC = ATC + cut / vol
        ATC = ATC / len(R)
        print('AB ' + str(AB) + " ATC " + str(ATC))


class qtcsGraph:

    def __init__(self, t_edges):
        self.edge = t_edges
        self.tadj_list, self.edge_stream,self.T = self.TemporalGraph()
        # print("number of nodes: " + str(len(self.tadj_list)))
        number, self.tmax = 0, 0
        for u in self.tadj_list:
            tset = set()
            number += len(self.tadj_list[u])
            for v in self.tadj_list[u]:
                for t in self.tadj_list[u][v]:
                    tset.add(t)
            if len(tset) > self.tmax:
                self.tmax = len(tset)
        # print("number of static edges: " + str(number / 2))
        # print("number of temporal edges: " + str(len(self.edge_stream) / 2))
        # print("number of timestamp: "+str(self.T))
        # print("self.tmax:" + str(self.tmax))
        self.number_temporal_edge=len(self.edge_stream) / 2
        self.ttp, self.dangling_state, self.t_vertex,self.number_t_vertex = self.Ttp()



    def TemporalGraph(self):
        tadj_list, temp = {}, set()
        # print( " is loading...")
        starttime = time.time()
        for e in self.edge:
                from_id, to_id, time_id = int(e[0]), int(e[1]), int(e[2])

                if from_id == to_id:
                    continue
                for (f_id, t_id) in [(from_id, to_id), (to_id, from_id)]:
                    temp.add((f_id, t_id, time_id))
        temp = list(temp)
        temp.sort(key=lambda x: x[2])
        # print([(temp[0][0], temp[0][1], 1)])
        edge_stream = [(temp[0][0], temp[0][1], 1)]
        t_index, t_current = 1, temp[0][2]
        for i in range(1, len(temp)):
            if temp[i][2] != t_current:
                t_index += 1
                t_current = temp[i][2]
            edge_stream.append((temp[i][0], temp[i][1], t_index))
        for f_id, t_id, time_id in edge_stream:
            if f_id in tadj_list:
                if t_id in tadj_list[f_id]:
                    tadj_list[f_id][t_id].add(time_id)
                else:
                    tadj_list[f_id][t_id] = {time_id}
            else:
                tadj_list[f_id] = {}
                tadj_list[f_id][t_id] = {time_id}
        endtime = time.time()
        # print("loading_graph_time(s)" + str(endtime - starttime))
        return (tadj_list, edge_stream,t_index)

    def Ttp(self):
        starttime = time.time()
        ttp, tnode_out_adj_list = {}, {}
        dangling_state = set()
        for (u, v, t) in self.edge_stream:
            if u in tnode_out_adj_list:
                for t1 in tnode_out_adj_list[u]:
                    if t1 == t:
                        continue
                    tnode_out_adj_list[u][t1].add((v, t))
            if v in tnode_out_adj_list:
                tnode_out_adj_list[v][t] = set()
            else:
                tnode_out_adj_list[v] = {}
                tnode_out_adj_list[v][t] = set()
        for v in tnode_out_adj_list:
            for t in tnode_out_adj_list[v]:
                ttp[(v, t)] = {}
                if len(tnode_out_adj_list[v][t]) == 0:
                    ttp[(v, t)][(v, t)] = 1
                    dangling_state.add((v, t))
                    continue
                sum1 = 0
                for (u, t1) in tnode_out_adj_list[v][t]:
                    sum1 = sum1 + self.f(t1 - t)
                for (u, t1) in tnode_out_adj_list[v][t]:
                    ttp[(v, t)][(u, t1)] = (self.f(t1 - t)) / sum1
        endtime = time.time()
        # print("compute_ttp_time(s)" + str(endtime - starttime))

        t_vertex = {}
        number_t_vertex=0

        for u in self.tadj_list:
            for v in self.tadj_list[u]:
                for t in self.tadj_list[u][v]:
                    if u not in t_vertex:
                        t_vertex[u] = {t}
                    else:
                        t_vertex[u].add(t)
            number_t_vertex+=len(t_vertex[u])

        return ttp, dangling_state, t_vertex,number_t_vertex

    def f(self, x):
        return 1 / x

    def core_decompisition(self):
        deg, core_number, core_renumber = {}, {}, {}
        max_core = 0
        myMinHeap = MinHeap([])
        n = len(self.tadj_list)
        n_core = 0
        for u in self.tadj_list:
            deg[u] = len(self.tadj_list[u])
            myMinHeap.insert([u, deg[u]])
        starttime = time.time()
        while n_core != n:
            x = myMinHeap.remove()
            if x[1] > max_core:
                max_core = x[1]
            core_number[x[0]] = max_core
            n_core += 1
            if core_number[x[0]] in core_renumber:
                core_renumber[core_number[x[0]]].add(x[0])
            else:
                core_renumber[core_number[x[0]]] = {x[0]}
            for u in self.tadj_list[x[0]]:
                if u not in core_number:
                    deg[u] = deg[u] - 1
                    myMinHeap.decrease_key(u, deg[u])
        endtime = time.time()
        # print("core_decomposition_time(s)" + str(endtime - starttime))
        return core_number, core_renumber

    def maintain_connected(self, temp, seed):
        q, visited = [seed], {seed}
        while q:
            v = q.pop()
            for u in self.tadj_list[v]:
                if u in temp and u not in visited:
                    q.append(u)
                    visited.add(u)
        return visited



    def Compute_tppr(self, alpha, seed):
        starttime = time.time()

        tppr, D = {}, defaultdict(lambda: defaultdict(int))

        e_out_seed = 0
        for u in self.tadj_list[seed]:
            e_out_seed += len(self.tadj_list[seed][u])
        for (u, v, t) in self.edge_stream:
            for t1 in D[u]:
                if (v,t) in self.ttp[(u,t1)]:
                    D[v][t] += (1 - alpha) * D[u][t1] * self.ttp[(u, t1)][(v, t)]
            if u == seed:
                D[v][t] = D[v][t] + (alpha) / e_out_seed

        for v in D:
            tppr[v] = 0
            for t in D[v]:
                if (v, t) in self.dangling_state:
                    D[v][t] = D[v][t] / (alpha)
                tppr[v] = tppr[v] + D[v][t]
        endtime = time.time()
        return tppr, endtime - starttime


    def qtcs_baseline(self, alpha, seed, k):
        starttime = time.time()
        q = []
        D = set()
        deg = {}
        for node_id in self.tadj_list:
            deg[node_id] = len(self.tadj_list[node_id])
            if deg[node_id] < k:
                q.append(node_id)
        while q:
            v = q.pop()
            D.add(v)
            for w in self.tadj_list[v]:
                if deg[w] >= k:
                    deg[w] = deg[w] - 1
                    if deg[w] < k:
                        q.append(w)
        kcore = set(self.tadj_list.keys()) - D
        if seed not in kcore:
            print("noanswer")
            return set(),0
        temp = self.maintain_connected(kcore, seed)
        tppr,time1 = self.Compute_tppr(alpha, seed)
        mymin_heap = MinHeap([])
        for u in temp:
            mymin_heap.insert([u, tppr[u]])
        D, best_indx = [], 0
        while mymin_heap.heap:
            u = mymin_heap.remove()[0]
            if u == seed:
                break
            if deg[u] < k:
                continue
            q = [u]
            while q:
                u = q.pop()
                D.append(u)
                for w in self.tadj_list[u]:
                    if deg[w] >= k:
                        deg[w] = deg[w] - 1
                        if deg[w] < k:
                            q.append(w)
                            if w == seed:
                                q = []
                                mymin_heap = MinHeap([])
                                break

            if mymin_heap.heap:
                best_indx = len(D)
        R = temp - set(D[:best_indx])
        result = self.maintain_connected(R, seed)
        endtime = time.time()
        return result, endtime - starttime


    def EGR(self, alpha, seed):
        starttime = time.time()

        tppr,time_tppr = self.Compute_tppr(alpha, seed)
        rho = {}
        mymin_heap = MinHeap([])
        for u in self.tadj_list:
            rho[u] = 0
            for v in self.tadj_list[u]:
                rho[u] = rho[u] + tppr[v]
            mymin_heap.insert([u, rho[u]])
        temp = set(self.tadj_list)
        opt = (mymin_heap.peek())[1]
        D, best_index = [], 0
        while temp:
            while (mymin_heap.peek())[1] <= opt:
                u = mymin_heap.remove()[0]
                temp.remove(u)
                D.append(u)
                if str(u) == str(seed):
                    temp = set()
                    break
                for v in self.tadj_list[u]:
                    if v in temp:
                        rho[v] = rho[v] - tppr[u]
                        mymin_heap.decrease_key(v, rho[v])
            if temp:
                opt = (mymin_heap.peek())[1]
                best_index = len(D)
        R = set(self.tadj_list) - set(D[:best_index])
        result = self.maintain_connected(R, seed)
        endtime = time.time()
        return result,time_tppr,endtime - starttime




    def propagation(self,v,t,alpha,C):
        if (v, t) not in self.dangling_state:
            for (w, t1) in self.ttp[(v, t)]:
                if (w, t1) not in self.r:
                    self.r[(w, t1)] = 0
                self.r[(w, t1)] = self.r[(w, t1)] + (1 - alpha) * self.r[(v, t)] * self.ttp[(v, t)][(w, t1)]
            if v not in self.tppr:
                self.tppr[v] = 0
            self.tppr[v] = self.tppr[v] + alpha * self.r[(v, t)]

            #maintain some heap structures
            if v in self.Q.heap_dict:
                self.sum_Q += alpha * self.r[(v, t)]
            if v in C:
                for w in self.tadj_list[v]:
                    if w in C:
                        self.inter_rho[w] = self.inter_rho[w] + alpha * self.r[(v, t)]
                        self.inter_rho_min_heap.increase_key(w, self.inter_rho[w])
                    if w in self.Q.heap_dict:
                        self.Q_with_C[w] = self.Q_with_C[w] + alpha * self.r[(v, t)]
                        self.Q.increase_key(w, self.Q_with_C[w])
            self.r_sum = self.r_sum - alpha * self.r[(v, t)]
            # maintain some heap structures

            self.r[(v, t)] = 0

        if (v, t) in self.dangling_state:
            if v not in self.tppr:
                self.tppr[v] = 0
            self.tppr[v] = self.tppr[v] + self.r[(v, t)]

            #maintain some heap structures
            if v in self.Q.heap_dict:
                self.sum_Q += self.r[(v, t)]
            if v in C:
                for w in self.tadj_list[v]:
                    if w in C:
                        self.inter_rho[w] = self.inter_rho[w] +self.r[(v, t)]
                        self.inter_rho_min_heap.increase_key(w, self.inter_rho[w])
                    if w in self.Q.heap_dict:
                        self.Q_with_C[w] = self.Q_with_C[w] + self.r[(v, t)]
                        self.Q.increase_key(w, self.Q_with_C[w])
            self.r_sum = self.r_sum -  self.r[(v, t)]
            # maintain some heap structures

            self.r[(v, t)] = 0


    def ALS(self, alpha, seed):
        starttime = time.time()
        self.r = {}
        e_out_seed = 0
        for v in self.tadj_list[seed]:
            e_out_seed = e_out_seed + len(self.tadj_list[seed][v])
        for v in self.tadj_list[seed]:
            for t in self.tadj_list[seed][v]:
                self.r[(v, t)] = 1 / e_out_seed

        self.inter_rho, self.inter_rho_min_heap = {}, MinHeap([])
        self.tppr = {}
        self.Q = MaxHeap([])
        self.Q.insert([seed, 0])
        self.Q_with_C = {}

        best, C, D = 0, set(), {seed}

        self.r_sum = 1
        unqualified = set()
        self.sum_Q = 0

        while self.Q.heap:
            u = self.Q.remove()[0]
            if u in self.tppr:
                self.sum_Q -= self.tppr[u]

            for v in self.tadj_list[u]:
                for t in self.t_vertex[v]:
                    if (v, t) in self.r and self.r[(v, t)] >1 / self.number_t_vertex:
                        self.propagation(v,t,alpha,C)

            #maintain some heap structures
            self.inter_rho[u] = 0
            for w in self.tadj_list[u]:
                if w in C:
                    if w in self.tppr:
                        self.inter_rho[u] = self.inter_rho[u] + self.tppr[w]
                    if u in self.tppr:
                        self.inter_rho[w] = self.inter_rho[w] + self.tppr[u]
                        self.inter_rho_min_heap.increase_key(w, self.inter_rho[w])
                if w in self.Q.heap_dict and u in self.tppr:
                    self.Q_with_C[w] = self.Q_with_C[w] + self.tppr[u]
                    self.Q.increase_key(w, self.Q_with_C[w])
            self.inter_rho_min_heap.insert([u, self.inter_rho[u]])
            # maintain some heap structures

            C.add(u)
            if self.inter_rho_min_heap.peek()[1] > best:
                best = self.inter_rho_min_heap.peek()[1]

            if self.r_sum<0: #self.r_sum may be negative because of the accuracy of the computer
                self.r_sum=0

            for v in self.tadj_list[u]:
                if v not in D:
                    D.add(v)
                    xv = self.r_sum
                    for w in self.tadj_list[v]:
                        if w not in unqualified and w in self.tppr:
                            xv = xv + self.tppr[w]
                    if xv >= best:
                        #maintain some heap structures
                        if v not in self.tppr:
                            self.tppr[v] = 0
                        self.sum_Q += self.tppr[v]
                        self.Q_with_C[v] = 0
                        for w in self.tadj_list[v]:
                            if w in C and w in self.tppr:
                                self.Q_with_C[v] = self.Q_with_C[v] + self.tppr[w]
                        # maintain some heap structures

                        self.Q.insert([v, self.Q_with_C[v]])
                    else:
                        unqualified.add(v)
            # min_inter_C = float('inf')
            # for v in C:
            #     inter_C = 0
            #     for w in self.tadj_list[v]:
            #         if w in C:
            #             inter_C += self.tppr[w]
            #     if inter_C < min_inter_C:
            #         min_inter_C = inter_C
            #
            # if format(min_inter_C, ".6f") != format(self.inter_rho_min_heap.peek()[1], ".6f"):
            #     print(min_inter_C)
            #     print(self.inter_rho_min_heap.peek()[1])
            #     print("wrong.....................")
            # Verify the correctness of the expanding algorithm and heap structure maintenance

            if self.sum_Q + self.r_sum < best:
                # print("Q prune is effective")
                # print("Q(len)"+str(len(self.Q.heap_dict)))
                for v in self.Q.heap_dict:
                    C.add(v)
                break

        endtime = time.time()
        expanding_time= endtime - starttime

        #Verify the correctness of the expanding algorithm and heap structure maintenance
        # print("self.r_sum" + str(self.r_sum))
        # print("rsum" + str(sum(self.r.values())))
        # vaiable = sum(self.tppr.values())+ self.r_sum
        # print("vaiable" + str(vaiable))
        # sum1 = 0
        # for u in self.Q.heap_dict:
        #     sum1 += self.tppr[u]
        # if format(sum1, ".8f") != format(self.sum_Q, ".8f"):
        #     print("QQQQQQQQQQQQwrong")
        #     print(sum1)
        #     print(self.sum_Q)
        starttime = time.time()
        R, rho_hat, flag = C.copy(), {}, True
        max_rho, min_rho = 0, float('inf')
        for u in C:
            rho_hat[u] = 0
            if u not in self.tppr:
                self.tppr[u] = 0
            for v in self.tadj_list[u]:
                if v in C  and v in self.tppr:
                    rho_hat[u] += self.tppr[v]
            if rho_hat[u] > max_rho:
                max_rho = rho_hat[u]
            if rho_hat[u] < min_rho and rho_hat[u] != 0:
                min_rho = rho_hat[u]
        temp = self.r_sum + max_rho
        epsion = temp / min_rho
        lambda_1 = epsion
        while flag:
            D, Q = set(), []
            for u in R:
                if epsion * rho_hat[u] <= temp:
                    Q.append(u)
                    if u == seed:
                        flag = False
                        Q = []
                        break
            while Q:
                u = Q.pop()
                D.add(u)
                for v in self.tadj_list[u]:
                    if v in R and v not in D:
                        rho_hat[v] = rho_hat[v] - self.tppr[u]
                        if epsion * rho_hat[v] <= temp:
                            Q.append(v)
                            if v == seed:
                                flag = False
                                Q = []
                                break
            if flag:
                lambda_1 = epsion
                R = R - D
                epsion = epsion / 2
        result = self.maintain_connected(R, seed)
        endtime = time.time()
        reducing_time= endtime - starttime
        return C,expanding_time,reducing_time,result,lambda_1


    def metric(self,S):
        temporal_edge_S=0
        time_S=set()
        for u in S:
            for v in self.tadj_list[u]:
                if v in S:
                    for t in self.tadj_list[u][v]:
                        time_S.add(t)
                        temporal_edge_S+=1
        TD=temporal_edge_S/(len(S)*(len(S)-1)*len(time_S))

        temporal_cut_S=0
        temporal_vol_S=0
        for u in S:
            for v in self.tadj_list[u]:
                if v not in S:
                    temporal_cut_S+=len(self.tadj_list[u][v])
            for w in self.tadj_list[u]:
                temporal_vol_S+=len(self.tadj_list[u][w])

        if temporal_vol_S>len(self.edge_stream)-temporal_vol_S:
            temporal_vol_S=len(self.edge_stream)-temporal_vol_S
        if temporal_vol_S==0:
            TC=1
        else:
            TC=temporal_cut_S/temporal_vol_S

        return TD, TC

    def t_vertex_sort(self): #temporal occurrence rank
        number_t_vertex=[]
        for u in self.tadj_list:
            number_t_vertex.append((u,len(self.t_vertex[u])))
        number_t_vertex.sort(key=lambda x:x[1])
        sorted_vertex=[] #sort vertex by increasing len(self.t_vertex[u]))
        for pair in number_t_vertex:
            sorted_vertex.append(pair[0])
        sorted_vertex_percent={}
        step=len(self.tadj_list)/10
        for i in range(1,11):
            sorted_vertex_percent[i]=[]
            for j in range(int((i-1)*step),int(i*step)):
                sorted_vertex_percent[i].append(sorted_vertex[j])

        return sorted_vertex_percent


    def inter_min_rho(self,H,alpha,seed): #for testing precision,recall and F1
        tppr,time_tppr = self.Compute_tppr(alpha, seed)
        min_inter_rho_H=float('inf')
        for u in H:
            inter_H= 0
            for v in self.tadj_list[u]:
                if v in H:
                    inter_H += tppr[v]
            if inter_H<min_inter_rho_H:
                min_inter_rho_H=inter_H

        return min_inter_rho_H

    def temporal_subgraph(self, S): #for case study
        interaction = {}
        for u in S:
            interaction[u] = {}
            for v in self.tadj_list[u]:
                if v in S:
                    interaction[u][v] = set()
                    for t in self.tadj_list[u][v]:
                        interaction[u][v].add(t)
        return interaction



class MinHeap:

    def __init__(self, array):
        self.idx_of_element = {}
        self.heap_dict = {} #
        self.heap = self.build_heap(array)

    def __getitem__(self, key): #
        return self.heap_dict[key]

    def insert(self, node):
        self.heap.append(node)
        self.idx_of_element[node[0]] = len(self.heap) - 1
        self.heap_dict[node[0]] = node[1]  #
        self.sift_up(len(self.heap) - 1)

    def build_heap(self, array):
        lastIdx = len(array) - 1
        startFrom = (lastIdx - 1) // 2

        for idx, i in enumerate(array):
            self.idx_of_element[i[0]] = idx
            self.heap_dict[i[0]] = i[1]

        for i in range(startFrom, -1, -1):
            self.sift_down(i, array)
        return array

    def remove(self):
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self.idx_of_element[self.heap[0][0]], self.idx_of_element[self.heap[-1][0]] = (
            self.idx_of_element[self.heap[-1][0]],
            self.idx_of_element[self.heap[0][0]],
        )

        x = self.heap.pop()
        del self.idx_of_element[x[0]]
        del self.heap_dict[x[0]]
        self.sift_down(0, self.heap)
        return x

    def increase_key(self, name, newValue):
        assert (
                self.heap[self.idx_of_element[name]][1] <= newValue
        ), "newValue must be larges that current value"
        self.heap[self.idx_of_element[name]][1] = newValue
        self.heap_dict[name] = newValue  #
        self.sift_down(self.idx_of_element[name], self.heap)

    def decrease_key(self, name, newValue):
        assert (
                self.heap[self.idx_of_element[name]][1] >= newValue
        ), "newValue must be less that current value"
        self.heap[self.idx_of_element[name]][1] = newValue
        self.heap_dict[name] = newValue  #
        self.sift_up(self.idx_of_element[name])

    # this is min-heapify method
    def sift_down(self, idx, array):
        while True:
            l = idx * 2 + 1
            r = idx * 2 + 2

            smallest = idx
            if l < len(array) and array[l][1] < array[idx][1]:
                smallest = l
            if r < len(array) and array[r][1] < array[smallest][1]:
                smallest = r

            if smallest != idx:
                array[idx], array[smallest] = array[smallest], array[idx]
                (
                    self.idx_of_element[array[idx][0]],
                    self.idx_of_element[array[smallest][0]],
                ) = (
                    self.idx_of_element[array[smallest][0]],
                    self.idx_of_element[array[idx][0]],
                )
                idx = smallest
            else:
                break

    def sift_up(self, idx):
        p = (idx - 1) // 2
        while p >= 0 and self.heap[p][1] > self.heap[idx][1]:
            self.heap[p], self.heap[idx] = self.heap[idx], self.heap[p]
            self.idx_of_element[self.heap[p][0]], self.idx_of_element[self.heap[idx][0]] = (
                self.idx_of_element[self.heap[idx][0]],
                self.idx_of_element[self.heap[p][0]],
            )  #
            idx = p
            p = (idx - 1) // 2

    def peek(self):
        return self.heap[0]


class MaxHeap:

    def __init__(self, array):
        self.idx_of_element = {}
        self.heap_dict = {} #
        self.heap = self.build_heap(array)
    #def __getitem__(self, key): #
    #    return self.heap_dict[key]
    def insert(self, node):
        self.heap.append(node)
        self.idx_of_element[node[0]] = len(self.heap) - 1
        self.heap_dict[node[0]] = node[1] #
        self.sift_up(len(self.heap) - 1)

    def build_heap(self, array):
        lastIdx = len(array) - 1
        startFrom = (lastIdx - 1) // 2

        for idx, i in enumerate(array):
            self.idx_of_element[i[0]] = idx
            self.heap_dict[i[0]] = i[1]

        for i in range(startFrom, -1, -1):
            self.sift_down(i, array)
        return array

    def remove(self):
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        self.idx_of_element[self.heap[0][0]], self.idx_of_element[self.heap[-1][0]] = (
            self.idx_of_element[self.heap[-1][0]],
            self.idx_of_element[self.heap[0][0]],
        )

        x = self.heap.pop()
        del self.idx_of_element[x[0]]
        del self.heap_dict[x[0]]
        self.sift_down(0, self.heap)
        return x

    def increase_key(self, name, newValue):
        assert (
                self.heap[self.idx_of_element[name]][1] <= newValue
        ), "newValue must be less that current value"
        self.heap[self.idx_of_element[name]][1] = newValue
        self.heap_dict[name] = newValue  #
        self.sift_up(self.idx_of_element[name])

    def decrease_key(self, name, newValue):
        assert (
                self.heap[self.idx_of_element[name]][1] >= newValue
        ), "newValue must be less that current value"
        self.heap[self.idx_of_element[name]][1] = newValue
        self.heap_dict[name] = newValue #
        self.sift_down(self.idx_of_element[name], self.heap)

    # this is max-heapify method
    def sift_down(self, idx, array):
        while True:
            l = idx * 2 + 1
            r = idx * 2 + 2

            largest = idx
            if l < len(array) and array[l][1] > array[idx][1]:
                largest = l
            if r < len(array) and array[r][1] > array[largest][1]:
                largest = r

            if largest != idx:
                array[idx], array[largest] = array[largest], array[idx]
                (
                    self.idx_of_element[array[idx][0]],
                    self.idx_of_element[array[largest][0]],
                ) = (
                    self.idx_of_element[array[largest][0]],
                    self.idx_of_element[array[idx][0]],
                )
                idx = largest
            else:
                break

    def sift_up(self, idx):
        p = (idx - 1) // 2
        while p >= 0 and self.heap[p][1] < self.heap[idx][1]:
            self.heap[p], self.heap[idx] = self.heap[idx], self.heap[p]
            self.idx_of_element[self.heap[p][0]], self.idx_of_element[self.heap[idx][0]] = (
                self.idx_of_element[self.heap[idx][0]],
                self.idx_of_element[self.heap[p][0]],
            )  #
            idx = p
            p = (idx - 1) // 2

    def peek(self):
        return self.heap[0]
