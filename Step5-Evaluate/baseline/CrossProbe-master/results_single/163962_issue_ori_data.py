# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import tensorly as tl
tl.set_backend("pytorch")
from tensorly.decomposition import parafac

x = torch.ones(12,3,12).to("mps")
print(x.shape)

weights, factors = parafac(x.detach(), 12, init="random", tol=1e-6)

a, m, b = factors