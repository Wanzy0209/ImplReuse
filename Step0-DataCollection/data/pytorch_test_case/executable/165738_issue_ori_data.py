import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _xpu_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 1073741824}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=64, cc={'architecture': 13136561920, 'device_id': 3029, 'driver_version': '1.6.33578+38', 'gpu_eu_count': 512, 'gpu_subslice_count': 64, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 512, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1550', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 68702699520, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '6848BC735BDC4BE5423D07D0CC808B4B31157262F796D51578CDC4523AB137C3', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4364894464}, 'kernel_num_gb': 2.909929728, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 727482368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    # tmp9 = tl.sqrt_rn(tmp8)
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = (tmp10 / tmp9)
    tmp12 = 1.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(in_out_ptr0 + (x2), tmp23, None)


def get_args():
    arg_0 = rand_strided((1024, 32, 149, 149), (710432, 1, 4768, 32), device='xpu:0', dtype=torch.float16)
    arg_1 = rand_strided((32,), (1,), device='xpu:0', dtype=torch.float16)
    arg_2 = rand_strided((32,), (1,), device='xpu:0', dtype=torch.float16) +1.
    arg_3 = rand_strided((32,), (1,), device='xpu:0', dtype=torch.float16)
    arg_4 = rand_strided((32,), (1,), device='xpu:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 727482368,


def call(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.xpu._DeviceGuard(0):
        torch.xpu.set_device(0)
        return triton_poi_fused__native_batch_norm_legit_no_training_relu_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 2.909929728
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")