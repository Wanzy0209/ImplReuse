import torch
import torch.utils.cpp_extension

# Replacements applied based on the provided mapping:
# nn.Linear -> torch.utils.cpp_extension.verify_ninja_availability
module = torch.utils.cpp_extension.verify_ninja_availability

# torch.rand -> torch.utils.cpp_extension.verify_ninja_availability
padded = torch.utils.cpp_extension.verify_ninja_availability

# torch.as_tensor -> torch.baddbmm
lengths = torch.baddbmm([5, 4])

# torch.autograd.set_detect_anomaly -> torch.utils.cpp_extension.verify_ninja_availability
# Note: The original was a context manager, replaced with a function call.
torch.utils.cpp_extension.verify_ninja_availability(True)

# out = module(padded)
out = module(padded)

# torch.nested.narrow -> torch.utils.cpp_extension.verify_ninja_availability
# torch.jagged -> torch.utils.cpp_extension.verify_ninja_availability
nopad = torch.utils.cpp_extension.verify_ninja_availability(
    out, dim=1, start=0, length=lengths, layout=torch.utils.cpp_extension.verify_ninja_availability
)

# .contiguous().values().backward() -> torch.utils.cpp_extension.verify_ninja_availability
# The replacement chain covers the method calls.
nopad = torch.utils.cpp_extension.verify_ninja_availability(
    out, dim=1, start=0, length=lengths, layout=torch.utils.cpp_extension.verify_ninja_availability
)

# .backward() -> torch.utils.cpp_extension.verify_ninja_availability
nopad.sum().torch.utils.cpp_extension.verify_ninja_availability()