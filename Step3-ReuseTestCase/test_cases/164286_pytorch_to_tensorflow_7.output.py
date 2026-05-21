import os
import tensorflow as tf

# Setup: Configure the environment variable required by tf.experimental.dtensor.jobs
# We use BNS-style names to trigger the specific sorting/validation logic found in the API,
# similar to how the original PyTorch test used sparse tensors to trigger layout logic.
os.environ["DTENSOR_JOBS"] = "/bns/worker2,/bns/worker1,/bns/worker0"

def f():
    """
    Wrapper function to test the API behavior.
    Mimics the structure of the original test case's function 'f'.
    """
    # In the original test, layout was printed. Here we print the job list.
    jobs_list = tf.experimental.dtensor.jobs()
    print(jobs_list)
    return jobs_list

# 1. Direct call (equivalent to print(f(a, x)) in the original test)
print("Direct call:")
result_direct = f()

# 2. Wrapped call (equivalent to vjp(f, a, x) in the original test)
# In TensorFlow, tf.function is the standard mechanism for transforming/executing functions
# in a different context (Graph mode vs Eager mode), which serves as an analog to the
# functional transformation context in the PyTorch bug report.
print("Wrapped call (tf.function):")
f_tf = tf.function(f)
result_wrapped = f_tf()

# Verify behavior
# The API implementation sorts BNS style names, so we expect the sorted list.
expected_jobs = ["/bns/worker0", "/bns/worker1", "/bns/worker2"]

assert result_direct == expected_jobs, f"Direct call failed: expected {expected_jobs}, got {result_direct}"
assert result_wrapped == expected_jobs, f"Wrapped call failed: expected {expected_jobs}, got {result_wrapped}"
assert result_direct == result_wrapped, "Results differ between direct and wrapped calls"

print("Test passed.")