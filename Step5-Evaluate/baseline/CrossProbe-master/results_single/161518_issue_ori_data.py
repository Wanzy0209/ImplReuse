# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import triton

for dtype in [torch.float8_e4m3fn, torch.bfloat16, torch.float32]:
    element_size = dtype.itemsize

    input1 = torch.zeros((128 // element_size) * 1024, 1024, device="cuda", dtype=dtype) # 128 MiB data
    input2 = torch.zeros((128 // element_size) * 1024, 1024, device="cuda", dtype=dtype) # 128 MiB data
    output = torch.zeros((128 // element_size) * 1024, 2048, device="cuda", dtype=dtype) # 256 MiB data

    torch.cat([input1, input2], dim=1, out=output) # 512 MiB IO

    output = triton.testing.do_bench_cudagraph(lambda: torch.cat([input1, input2], dim=1, out=output), rep=100)

    bdwidth = 512 / output * 1000 / 1000 / 1000 # TiB/s

    print("\t".join([str(dtype), f"{output:.2f} ms", f"{bdwidth:.2f} TiB/s"]))