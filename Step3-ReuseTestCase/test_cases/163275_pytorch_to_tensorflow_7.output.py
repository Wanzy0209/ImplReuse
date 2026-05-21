import os
import tensorflow as tf

# Setup: Configure the environment variable required by tf.experimental.dtensor.jobs
# The API reads from the "DTENSOR_JOBS" environment variable.
os.environ["DTENSOR_JOBS"] = "worker0,worker1,worker2"

# Adaptation: Define a wrapper function to mimic the structure of the original test case.
# The original test used a compiled function; here we simply wrap the API call.
def get_cluster_jobs():
    return tf.experimental.dtensor.jobs()

# Execution: Call the function
jobs = get_cluster_jobs()

# Verification: Assert the expected behavior based on the API implementation
# The API should split the environment variable string by commas.
assert jobs == ["worker0", "worker1", "worker2"], f"Expected ['worker0', 'worker1', 'worker2'], got {jobs}"

# Additional Verification: Test the BNS validation logic mentioned in the API source code
# If job names start with "/bns/", they must be sorted, or a ValueError is raised.
os.environ["DTENSOR_JOBS"] = "/bns/worker1,/bns/worker0"

try:
    get_cluster_jobs()
    # If we reach here, the validation logic failed to raise an error
    raise AssertionError("Expected ValueError for unsorted BNS style job names")
except ValueError as e:
    # Check if the error message matches the implementation
    assert "Sort entries" in str(e)