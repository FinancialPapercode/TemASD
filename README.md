# The code for  TempASD

This repo contains the source code for the TempASD model.

## Programming language

**Pytorch** with version 1.8.0 or later

## Required libraries

0. You need to set up the environment for running the experiments (Python 3.7 or above)

1. Install **Pytorch** with version 1.8.0 or later

2. Install **torch-geometric** package with version 2.0.1

   Note that it may need to appropriately install the package `torch-geometric` based on the CUDA version (or CPU version if GPU is not available). Please refer to the official website https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for more information on installing prerequisites.

   For example (Mac / CPU)

   ```
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
   ```

3. Install **Deepsnap** with version 0.2.1

   ```
   pip install deepsnap
   ```

## Hardware info

NVIDIA GeForce RTX 4090 24G GPU

Intel Xeon Platinum 8370C processor 32-core 2.8G CPU

## Datasets info

### Ethereum Blockchain

The ETH datasets are data from the Ethereum blockchain shared by Google BigQuery[1]. They contain token transactions from source to sink addresses with the transaction amount. The ETH datasets are also used in the SOTA work AntiBenford [2].

Our datasets are available on Google Drive: https://drive.google.com/drive/folders/1TOeVzgQJAYdlGhaabtGcg8WDE9e73l3z?usp=sharing 

- dataset/eth-2018jan
- dataset/eth-2019jan

Each of them contains a ground-truth subgraphs file `{name}-1.90.anomaly.txt` and an edge file `{name}-1.90.ungraph.txt`.

### Blur

The Blur dataset is a transaction dataset from the NFT marketplace [4], which is compiled by Etherscan API [4] from Oct. 19, 2022 to Apr. 1, 2023. They contain NFT transactions among addresses.

dataset/blur contains a file with abnormal subgraphs `{name}-1.90.anomaly.txt` and an edge file `{name}-1.90.ungraph.txt`.

### PlusTokenPonzi

The PlusTokenPonzi dataset [5] is a real Ethereum blockchain transaction dataset involving money laundering activities from EthereumHeist. It includes detailed information such as transaction timestamps and amounts, service provider address labels, hierarchical levels of laundering addresses, etc. This is the first publicly available dataset with ground-truth data on money laundering.

dataset/PlusTokenPonzi contains a file with abnormal subgraphs {name}-1.90.anomaly.txt and an edge file {name}-1.90.ungraph.txt

## Data preprocessing

Consistent with AntiBenford [2], transactions valued at less than 1 unit are excluded during preprocessing. The primary step involves constructing the graph. To represent multiple transactions between two nodes, we combine them into a single edge. Each edge is associated with the transaction frequency and the financial distribution, i.e., the probabilities of transaction amounts starting with a certain digit. This resulting input graph is used for our analysis.

The edge file requires pre-processing. The following steps need to be taken:

- The pre-processing details are in /dataset/eth-2018jan/sum.py and /dataset/eth-2019jan/sum.py.

## How to run the code for  TempASD

```
python run.py --dataset=eth-2019jan
```

Main arguments:

```
--dataset [eth-2019jan,eth-2018jan,blur]: the dataset to run
--n_layers: number of GNN layers
--pred_size: total number of predicted anomaly subgraphs
--agent_lr: the learning rate of Anomalous Subgraph Refinement
```

  For more argument options, please refer to `run.py`  

## References

[1]https://www.kaggle.com/bigquery/ethereum-blockchain

[2]T. Chen and C. Tsourakakis, "Antibenford subgraphs: Unsupervised anomaly detection in financial networks", (KDD2022)

[3]Z.Chenyu, C.Hongzhou, W.Hao, Z.Junyu, and C Wei, "ARTEMIS: Detecting Airdrop Hunters in NFT Markets with a Graph Learning System", (WWW2024)

[4]https://etherscan.io/

[5]https://github.com/lindan113/EthereumHeist?tab=readme-ov-file
