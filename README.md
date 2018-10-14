# GGNN_Reasoning
This is a Pytorch implementantion of [Gated Graph Neural Network](https://arxiv.org/pdf/1511.05493.pdf) (Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015).). This implementation follows the framework of
[JamesChuanggg/ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch). The main difference is that my implemantation are more suitable for graph datasets with tremendous edge types such as Knowledge Graphs as it's more memory-efficient.

Though our scenario is using GGNN to approximate the ABox consistency checking problem in OWL EL, where each ABox can be sample can be deemed as a small directed graph, the implementation is quite generic.

For the input json data format:
each sample has the format as follows,
{'target': label of sampe, 
'graph': all edges in the graph, each edge is represented as a triple: (source_id, edge_id, target_id),
'node_features:' task-specific innitial annotation for each node in the graph
}
(All ids start from 1)

To run the code, please use command **'python main.py`**.
