#%%
import dgl
from dgl import DGLGraph
from dgl.data import RedditDataset
data = RedditDataset()

# %%
g = data[0]
num_class = data.num_classes
feat = g.ndata['feat']  # get node feature
label = g.ndata['label']

# %%
set(label.tolist())
# %%
