# from torch.nn import *
# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


#     def forward(self, add_1, mul_5, add_3, mul_9, episode_builder_position_encoding_observations_weight, slice_1, mul, mul_1, add_6, add_15, add_16, add_17, add_18, add_13):
#         _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default(slice_1, dtype = torch.uint8, device = device(type='cuda', index=0), layout = torch.strided);  slice_1 = _assert_tensor_metadata_default = None
#         _assert_tensor_metadata_default_1 = torch.ops.aten._assert_tensor_metadata.default(mul, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  mul = _assert_tensor_metadata_default_1 = None
#         _assert_tensor_metadata_default_2 = torch.ops.aten._assert_tensor_metadata.default(mul_1, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  mul_1 = _assert_tensor_metadata_default_2 = None
#         _assert_tensor_metadata_default_3 = torch.ops.aten._assert_tensor_metadata.default(add_1, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  add_1 = _assert_tensor_metadata_default_3 = None
#         _assert_tensor_metadata_default_4 = torch.ops.aten._assert_tensor_metadata.default(mul_5, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  mul_5 = _assert_tensor_metadata_default_4 = None
#         _assert_tensor_metadata_default_5 = torch.ops.aten._assert_tensor_metadata.default(add_3, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  add_3 = _assert_tensor_metadata_default_5 = None
#         _assert_tensor_metadata_default_6 = torch.ops.aten._assert_tensor_metadata.default(mul_9, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  mul_9 = _assert_tensor_metadata_default_6 = None
#         _assert_tensor_metadata_default_7 = torch.ops.aten._assert_tensor_metadata.default(add_6, dtype = torch.float32, device = device(type='cuda', index=0), layout = torch.strided);  add_6 = _assert_tensor_metadata_default_7 = None
#         arange_1 = torch.ops.aten.arange.start(180, 181, device = device(type='cuda', index=0), pin_memory = False)
#         add_14 = torch.ops.aten.add.Tensor(arange_1, 198);  arange_1 = None
#         stack_1 = torch.ops.aten.stack.default([add_13, add_14, add_15, add_16, add_17, add_18]);  add_13 = add_14 = add_15 = add_16 = add_17 = add_18 = None
#         select_13 = torch.ops.aten.select.int(stack_1, 0, 0);  stack_1 = None
#         embedding_11 = torch.ops.aten.embedding.default(episode_builder_position_encoding_observations_weight, select_13);  episode_builder_position_encoding_observations_weight = select_13 = None
#         return (embedding_11,)


import os

os.environ["TORCHINDUCTOR_FREEZING"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor"

import torch
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.allow_complex_guards_as_runtime_asserts = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.allow_rnn = False
torch._inductor.config.cpp_wrapper = False
torch._inductor.config.graph_partition = False
torch._inductor.config.unroll_reductions_threshold = 8
torch._inductor.config.generate_intermediate_hooks = True
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.autotune_cublasLt = True
torch._inductor.config.triton.autotune_at_compile_time = None
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.aot_inductor.output_path = ""
torch._inductor.config.aot_inductor.serialized_in_spec = ""
torch._inductor.config.aot_inductor.serialized_out_spec = ""
torch._inductor.config.aot_inductor.package = False
torch._inductor.config.aot_inductor.metadata = {"AOTI_DEVICE_KEY": "cuda"}
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False


isolate_fails_code_str = None


# torch version: 2.8.0+cu128
# torch cuda version: 12.8
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2025 NVIDIA Corporation
# Built on Fri_Feb_21_20:23:50_PST_2025
# Cuda compilation tools, release 12.8, V12.8.93
# Build cuda_12.8.r12.8/compiler.35583870_0

# GPU Hardware Info:
# NVIDIA TITAN V : 2

exported_program = torch.export.load("./exported_program.pt2")
# print(exported_program.graph)
config_patches = {"aot_inductor.package": True}
if __name__ == "__main__":
    from torch._dynamo.repro.aoti import run_repro

    with torch.no_grad():
        run_repro(
            exported_program,
            config_patches=config_patches,
            accuracy="",
            command="run",
            save_dir="./checkpoints",
            check_str=None,
        )