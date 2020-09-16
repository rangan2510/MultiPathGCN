# Wide Graph Convolutional Networks
 
## Abstract

Graph convolution networks have recently garnered a lot of attention for representation learning on non-Euclidian feature spaces. Recent research has focused on stacking multiple layers like in convolutional neural networks for the increased expressive power of graph convolution networks. However, simply stacking multiple graph convolution layers lead to issues like vanishing gradient, over-fitting and over-smoothing. Such problems are much less when using shallower networks, even though the shallow networks have lower expressive power. In this work, we have proposed a novel Multipath Graph convolutional neural network that aggregates the output of multiple different shallow networks. The proposed network not only provides increased test accuracy but also require fewer epochs of training to converge while training on standard benchmark datasets for the task of node property prediction. 

## Introduction

Graph convolutional networks (GCNs) enable learning in non-Euclidean feature spaces, such as in the case of graphs and 3D point cloud data. The graph convolution operation is a generalization of the convolution operation used in convolution neural networks (CNNs). In the case of graphs, convolution is implemented using message passing. Information is passed to a node from its neighbours and the aggregated value is used to update the feature values. Recent works have shown that increasing the depth of these networks provide improved performance. However, naively stacking multiple layers also reduce the performance of these networks. Newer training techniques and aggregation functions that enable training of deeper networks do that at the expense of increased memory footprint and training times. This work explores the concept of multipath graph convolutional neural networks, where multiple networks of different depths are trained in parallel, each learning a different representation of the data. The outputs of the networks are finally aggregated in the final layers. The multiple parallel networks also provide the gradient alternate paths, providing faster convergence, while having the same number of trainable parameters. 
In this work, we test extensively compare the multipath graph convolutional networks with deep GCNs that are implemented by simply stacking multiple layers and also with deep GCNs with residual connections. 

## Contribution

## Results

## Discussion