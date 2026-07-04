# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

GB = 1024*1024*1024
device = torch.device('cuda')


def dump_mem():
    free, total = torch.cuda.mem_get_info()
    print(f"free: {free / (GB):.2f} GiB, total: {total / (GB):.2f} GiB")


def alloc_1gb():
    return torch.empty((GB,), dtype=torch.int8, device=device)


print("before allocation")
dump_mem()

t0 = alloc_1gb()
print("t0 allocated")
dump_mem()

mpool = torch.cuda.MemPool()
with torch.cuda.use_mem_pool(mpool):
    t1 = alloc_1gb()
    print("t1 allocated")
    dump_mem()

t0 = None
t1 = None
print("t0 and t1 freed")
dump_mem()
torch.cuda.empty_cache()
print("cache emptied")
dump_mem()