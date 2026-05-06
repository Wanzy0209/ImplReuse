2025-08-12T08:03:39.9724498Z FAILED: [code=1] caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl/torch_xpu_ops_gen_NanCheck_XPU.cpp.o /pytorch/build/caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl/torch_xpu_ops_gen_NanCheck_XPU.cpp.o 
2025-08-12T08:03:39.9728735Z cd /pytorch/build/caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl && /opt/_internal/cpython-3.10.18/lib/python3.10/site-packages/cmake/data/bin/cmake -E make_directory /pytorch/build/caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl/. && /opt/_internal/cpython-3.10.18/lib/python3.10/site-packages/cmake/data/bin/cmake -D verbose:BOOL=OFF -D generated_file:STRING=/pytorch/build/caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl/./torch_xpu_ops_gen_NanCheck_XPU.cpp.o -P /pytorch/build/caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl/torch_xpu_ops_gen_NanCheck_XPU.cpp.o.Release.cmake
2025-08-12T08:03:39.9732771Z In file included from /pytorch/third_party/torch-xpu-ops/src/xccl/NanCheck_XPU.cpp:7:
2025-08-12T08:03:39.9733500Z In file included from /pytorch/torch/csrc/api/include/torch/torch.h:3:
2025-08-12T08:03:39.9734118Z In file included from /pytorch/torch/csrc/api/include/torch/all.h:9:
2025-08-12T08:03:39.9734734Z In file included from /pytorch/torch/csrc/api/include/torch/data.h:3:
2025-08-12T08:03:39.9735415Z In file included from /pytorch/torch/csrc/api/include/torch/data/dataloader.h:3:
2025-08-12T08:03:39.9736203Z In file included from /pytorch/torch/csrc/api/include/torch/data/dataloader/stateful.h:4:
2025-08-12T08:03:39.9737038Z In file included from /pytorch/torch/csrc/api/include/torch/data/dataloader/base.h:3:
2025-08-12T08:03:39.9737854Z In file included from /pytorch/torch/csrc/api/include/torch/data/dataloader_options.h:4:
2025-08-12T08:03:39.9738957Z /pytorch/torch/csrc/api/include/torch/types.h:7:10: fatal error: 'torch/csrc/autograd/generated/variable_factories.h' file not found
2025-08-12T08:03:39.9739918Z     7 | #include <torch/csrc/autograd/generated/variable_factories.h>
2025-08-12T08:03:39.9740424Z       |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2025-08-12T08:03:39.9740804Z 1 error generated.
2025-08-12T08:03:39.9741355Z CMake Error at torch_xpu_ops_gen_NanCheck_XPU.cpp.o.Release.cmake:145 (message):
2025-08-12T08:03:39.9741874Z   Error generating file
2025-08-12T08:03:39.9742495Z   /pytorch/build/caffe2/aten_xpu/src/CMakeFiles/torch_xpu_ops.dir/xccl/./torch_xpu_ops_gen_NanCheck_XPU.cpp.o
2025-08-12T08:03:39.9743093Z