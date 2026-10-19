import torch
import torch.backends.cusparselt

# Helper function to make the test runnable, as make_custom_pool is not in the replacement list
# but is required by the original test case logic.
def make_custom_pool(x):
    return x

pool1 = make_custom_pool(1)

# Replaced torch.cuda.use_mem_pool and torch.cuda.MemPool with torch.backends.cusparselt.version
# Since torch.backends.cusparselt.version is not a context manager, we call it directly
# to preserve the logic of checking/initializing the backend state.
print("Pool 1 ctx start")
torch.backends.cusparselt.version()

# Replaced torch.randn with torch.var_mean
# torch.var_mean requires an input tensor, so we create one using torch.empty
# to ensure the test is runnable and valid.
x1 = torch.var_mean(torch.empty(8, device="cuda"))

print("Pool 1 ctx end")
del x1