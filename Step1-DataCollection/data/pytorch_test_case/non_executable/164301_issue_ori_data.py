(torch) [danvm@devgpu007.snb3 ~/ao/benchmarks/mx_formats (mx-a2a-2)]$ CUDA_VISIBLE_DEVICES=7 python cast_bench.py --mode dim0_mxfp8_floor
M 16384 K 16384 BLOCK_SIZE 32
GPU: NVIDIA B200
torch version: 2.10.0.[dev20250930](https://l.workplace.com/l.php?u=https%3A%2F%2Fwww.internalfb.com%2Fintern%2Fbunny%2F%3Fq%3Ddev20250930&h=AT3oNVz-GDTWsBu4uQV0JnkMocQLFmUI0vZFep5jacZOZsfQWFsUIunKLKusdxT9qpSa1BjidHVSCdH4ClhuNFQiqN7UJM_3tSoiRNyXK8YGz7Vd3FTiCoOhUeTFp4nT-72p4WKZTDnY7d8ok8i4Iw)+cu128
triton version: 3.5.0
mode: dim0_mxfp8_floor
time_us 547.7439761161804
mem_bw_gbps 1485.5388858305025