import torch

torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1215252001)

# Generate the specific tensors from the bug report
arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))
arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1))

# Adapted function using torch.all
def all_program(t1, t2, t3):
    # Using torch.all on the tensors with specific strides/shapes
    r1 = torch.all(t1)
    r2 = torch.all(t2)
    r3 = torch.all(t3)
    return r1, r2, r3

# Test Eager
result_eager = all_program(arg_0, arg_1, arg_2)
print(' eager success')

# Test Compiled
compiled_program = torch.compile(all_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(arg_0, arg_1, arg_2)
print(' compile success')

# Verify results match
assert result_eager == result_compiled
print(' results match')