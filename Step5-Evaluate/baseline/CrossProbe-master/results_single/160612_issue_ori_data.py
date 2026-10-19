# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

m = prune.random_unstructured(nn.Linear(5, 7), name = "weight", amount= 0.2)
m = prune.remove(m, name = "weight")
m