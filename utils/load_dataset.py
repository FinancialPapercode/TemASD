import networkx as nx
def load(name):
    """Load snap dataset"""
    abnormalsubgraphs = open(f"./dataset/{name}/{name}-1.90.anomaly.txt")
    edges = open(f"./dataset/{name}/{name}-1.90.weight.txt")

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
    return nodes, edges,abnormalsubgraphs

def loadTime(name):
    """Load snap dataset"""
    with open(f"./dataset/{name}/{name}-1.90.ungraph.txt", 'r') as file:
        edges = [[int(i) for i in line.split()] for line in file]

    nodes = set()
    edge_weights = []

    for u, v, timestamp in edges:
        nodes.add(u)
        nodes.add(v)
        edge_weights.append(timestamp)

    mapping = {node: i for i, node in enumerate(sorted(nodes))}
    edges = [[mapping[u], mapping[v], timestamp] for u, v, timestamp in edges]

    print(f"[{name.upper()}], #TNodes {len(nodes)}, #TEdges {len(edges)} ")

    return nodes, edges

def get_start_digit(v):
    if v==0:
        return 0
    if v<0:
        v = -v
    while v<1:
        v = v*10
    return int(str(v)[:1])


def load_benford():
    G = nx.Graph()
    f = open('./utils/jan18.csv', 'r')
    # f = open('./utils/eth-2019jan.csv', 'r')
    edgelist = []
    node_map = {}
    line = f.readline()
    line = f.readline()
    e_count = 0
    n_idx = 0
    while line:
        tmp = line.split(',')
        line = f.readline()
        date = tmp[-1].strip()
        money = tmp[-2].strip()
        if len(money) <= 18:
            continue
        if tmp[1] == tmp[2]:
            continue
        if tmp[1] not in node_map:
            node_map[tmp[1]] = n_idx
            n_idx += 1
        if tmp[2] not in node_map:
            node_map[tmp[2]] = n_idx
            n_idx += 1
        money = int(money[:-18])
        edgelist.append((node_map[tmp[1]], node_map[tmp[2]], get_start_digit(money), money, date))

        if node_map[tmp[1]] in G and node_map[tmp[2]] in G[node_map[tmp[1]]]:
            # 初始化权重为包含单个整数的列表
            if 'weight' not in G[node_map[tmp[1]]][node_map[tmp[2]]]:
                G[node_map[tmp[1]]][node_map[tmp[2]]]['weight'] = []
            # 将新的权重追加到现有列表中
            G[node_map[tmp[1]]][node_map[tmp[2]]]['weight'].append(get_start_digit(money))
        else:
            # 创建新的带有权重列表的边
            G.add_edge(node_map[tmp[1]], node_map[tmp[2]], weight=[get_start_digit(money)])
        e_count += 1
    return edgelist , G

def load_syn():
    G = nx.Graph()
    f = open('./utils/dblp-trans.csv', 'r')
    edgelist = []
    line = f.readline()
    line = f.readline()
    e_count = 0
    while line:
        tmp = line.split(',')
        line = f.readline()
        date = tmp[-2].strip()
        money = tmp[-1].strip()
        money = int(money)
        edgelist.append((int(tmp[1]), int(tmp[2]), get_start_digit(money), money, date))

        if int(tmp[1]) in G and int(tmp[2]) in G[int(tmp[1])]:
            # 初始化权重为包含单个整数的列表
            if 'weight' not in G[int(tmp[1])][int(tmp[2])]:
                G[int(tmp[1])][int(tmp[2])]['weight'] = []
            # 将新的权重追加到现有列表中
            G[int(tmp[1])][int(tmp[2])]['weight'].append(get_start_digit(money))
        else:
            # 创建新的带有权重列表的边
            G.add_edge(int(tmp[1]), int(tmp[2]), weight=[get_start_digit(money)])
        e_count += 1

    return edgelist, G
