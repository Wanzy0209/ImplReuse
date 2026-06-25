import os
os.environ["TORCHINDUCTOR_FREEZING"] = "1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torchinductor"

import torch
import torch._inductor.inductor_prims
import torch._dynamo.config
import torch._inductor.config

torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True

# Load exported program and run compilation
# exported_program = torch.load("exported_program.pt")
# torch._inductor.aoti_compile_and_package(exported_program)