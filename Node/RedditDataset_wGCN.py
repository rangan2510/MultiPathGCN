import random
import dgl
import dgl.function as fn
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GraphConv
import time
import numpy as np
import psutil
from statistics import mean
from dgl.data import RedditDataset
import networkx as nx

###############################################################################
# 

dgl.seed(0)
dgl.random.seed(0)

###############################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GraphConv(602, 128)
        self.layer2 = GraphConv(128, 128)
        self.layer3 = GraphConv(128, 128)
        self.fc = nn.Linear(128, 41)
    
    def forward(self, g, features):
        x = _x = F.relu(self.layer1(g, features))
        _x = F.relu(self.layer2(g, _x))
        x = F.relu(self.layer2(g, x))
        x = F.relu(self.layer3(g, x))
        x += _x
        x = F.relu(x)
        x = self.fc(x)
        return x
net = Net()
print(net)

###############################################################################
# We load the RedditDataset dataset using DGL's built-in data module.

def load_data():
    data = RedditDataset()
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    valid_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    return g, features, labels, train_mask, test_mask

###############################################################################
# Eval

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

###############################################################################
# Train
print("CPU cores:", psutil.cpu_count())
g, features, labels, train_mask, test_mask = load_data()
# Add edges between each node and itself to preserve old node representations
g.add_edges(g.nodes(), g.nodes())
print(g)
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []

best_score = 0
best_epoch = 0
for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)
    
    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
    if best_score < acc:
        best_score = acc
        best_epoch = epoch


print("Best Test Acc {:.4f} at Epoch {:05d}".format(best_score, best_epoch))