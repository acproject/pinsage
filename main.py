from herbRecommendation import HerbRecommendation
from pinsage import PinSage
import numpy as np
import time
import torch
import dgl
import torch.nn as nn
import torch.optim as optim
from utils import *

useCuda = False
randomseed = 42
lr = 0.001
weight_decay = 5e-4
feature_sizes = [20, 30]
T = 1
n_traces = 5
n_hops = 2
epochs = 200

cuda = useCuda and torch.cuda.is_available()
np.random.seed(randomseed)
torch.manual_seed(randomseed)
if cuda:
    torch.cuda.manual_seed(randomseed)

label = load_data('train.txt')
herb_rec = HerbRecommendation('')
g, symp_ids, herb_ids = herb_rec.todglgraph()
pin = PinSage(G=g,
              feature_sizes=feature_sizes,
              T=T,
              n_traces=n_traces,
              n_hops=n_hops)

optimizer = optim.Adam(pin.parameters(),
                       lr=lr,
                       weight_decay=weight_decay)

nodeset = symp_ids + herb_ids
print(len(symp_ids))
print(len(herb_ids))
print(len(nodeset))
features = torch.eye(len(nodeset), dtype=torch.float)

def train(epoch, model):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    h, sh = pin(features, nodeset)
    if epoch == epochs-1:
        savedP = sh.detach().numpy()
        np.savetxt('p.txt', savedP)

    loss = nn.BCEWithLogitsLoss()
    loss_train = loss(sh, label)
    loss_train.backward()
    optimizer.step()

    print(
        'Epoch: {:04d}'.format(epoch),
        'loss_train: {:.4f}'.format((loss_train).item()),
        'time: {:.4f}'.format(time.time() - t)
    )


time_total = time.time()
print('train start ...')
for epoch in range(epochs):
    train(epoch)

print('Total time elapsd: {:.4f}s'.format(time.time() - time_total))














