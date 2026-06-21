import torch
import functools
import sys

# Handle missing torch._inductor module gracefully
try:
    from torch._inductor import config
except ImportError:
    print("Skipping test: torch._inductor module not found.")
    sys.exit(0)

# Counter to verify execution count
pass_execution_count = 0

def custom_pre_pass(gm):
    """
    Custom pre-pass that increments a counter to track invocations.
    This simulates the user-defined pass mentioned in the bug report.
    """
    global pass_execution_count
    pass_execution_count += 1
    return gm

# Set the config to trigger the bug path
# The bug occurs when config.joint_custom_pre_pass is set, causing it to run twice.
config.joint_custom_pre_pass = custom_pre_pass

# Define a function using the similar API (functools.reduce)
# The extracted call chain points to torch._dynamo.polyfills.functools.reduce,
# which is the polyfill for the standard functools.reduce.
def reduce_fn(x):
    return functools.reduce(lambda a, b: a + b, x)

# Compile the function using torch.compile (Original API Under Test)
# The bug is in the inductor backend's joint_graph.py, which is triggered by compilation.
compiled_fn = torch.compile(reduce_fn, backend="inductor")

# Run the compiled function
input_data = [1, 2, 3, 4]
result = compiled_fn(input_data)

# The bug causes the pass to run twice (count == 2).
# The fix ensures it runs exactly once (count == 1).
assert pass_execution_count == 1, f"Expected pass to run once, but ran {pass_execution_count} times."

# Verify the result is correct
assert result == 10, f"Expected result 10, got {result}"

print("Test passed: joint_custom_pre_pass executed exactly once.")