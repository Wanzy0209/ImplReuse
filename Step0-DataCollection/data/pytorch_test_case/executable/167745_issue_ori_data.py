pool1 = make_custom_pool(1)

with torch.cuda.use_mem_pool(torch.cuda.MemPool(pool1)):
    print("Pool 1 ctx start")
    x1 = torch.randn(8, device="cuda")
    print("Pool 1 ctx end")
del x1