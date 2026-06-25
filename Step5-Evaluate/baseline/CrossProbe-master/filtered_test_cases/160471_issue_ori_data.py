import torch
import dis

def large_function():
    # Create a function that generates EXTENDED_ARG opcodes
    # by having many variables/constants
    v = [i for i in range(300)]
    return sum(v)

# Show the disassembly with EXTENDED_ARG
dis.dis(large_function.__code__)

# Run through TorchDynamo to see the issue
compiled = torch._dynamo.optimize()(large_function)
compiled()