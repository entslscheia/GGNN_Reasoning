# GGNN_Reasoning
This is a Pytorch implementantion of [Gated Graph Neural Network](https://arxiv.org/pdf/1511.05493.pdf) (Li, Yujia, et al. "Gated graph sequence neural networks." arXiv preprint arXiv:1511.05493 (2015).).
Though our scenario is to approximate the ABox consistency checking problem in OWL EL using GGNN, the implementation is quite generic.
Moreover, one of the features of this implementation is that it's more memory-efficient for graph dataset with trenmendous edge types such as Knowledge Graphs.

To run the code, just use command **'python main.py`**.
