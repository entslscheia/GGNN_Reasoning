# GGNN_Reasoning
This is a Pytorch implementantion of [Gated Graph Neural Network](https://arxiv.org/pdf/1511.05493.pdf) (Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015)). This implementation follows the framework of
[JamesChuanggg/ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch). The main difference is that my implemantation are more suitable for graph datasets with tremendous edge types such as Knowledge Graphs as it's more memory-efficient. Note that, **most other implementations you can find are designed for datasets with only several edge types, such as bABI dataset.**

Though our scenario is using GGNN to approximate the ABox consistency checking problem in OWL EL, where each ABox sample can be deemed as a small directed graph and thus the consistency checking can be modeled as a graph-level binary classification problem, the implementation is quite generic.

Usage:
- For the input json data format:<br>
each sample has the format as follows,<br>
**{'target': label of sample, <br>
'graph': all edges in the graph, each edge is represented as a triple: (source_id, edge_id, target_id), <br>
'node_features': task-specific innitial annotation for each node in the graph <br>
}**<br>
(All ids start from 1)
- To run the code, please use command **`python main.py`**.
