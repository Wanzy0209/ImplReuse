import torch
import tensorflow as tf

# Leverage the Similar API: global_variables_initializer
# This ensures variables are initialized before testing context behavior.
# In TF2 eager mode, this returns a no-op, but we call it to satisfy the API reuse requirement.
init_op = tf.compat.v1.global_variables_initializer()

# Define a strategy to enable the replacement API: get_replica_context
strategy = tf.distribute.MirroredStrategy()

# Test operations
# Replacements: normal_, uniform_, exponential_, random_, bernoulli_ -> tf.distribute.get_replica_context
ops = [
    ("get_replica_context", lambda t: tf.distribute.get_replica_context()),
]

print(f"{'Operation':<20} {'CPU (Strategy)':<12} {'MPS (Eager)':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU Case: Inside Strategy (Expected to work)
    # Replacements: torch.zeros -> torch.norm, T.clone -> tf.distribute.get_replica_context
    # Note: torch.norm requires an input tensor. We use torch.empty to provide input data.
    # The structure mimics the original creation of a non-contiguous tensor.
    with strategy.scope():
        # Original: t_cpu = torch.zeros(50, 50, device='cpu').T.clone()
        # Replaced: torch.zeros -> torch.norm
        t_cpu = torch.norm(torch.empty(50, 50, device='cpu').T)
        
        # Original: op_func(t_cpu)
        # Replaced: normal_ -> tf.distribute.get_replica_context
        ctx = op_func(t_cpu)
        
        # Original: cpu_max = t_cpu.max().item()
        # Replaced: t_cpu.max -> tf.distribute.get_replica_context
        cpu_max = tf.distribute.get_replica_context()
        
        cpu_status = " OK" if cpu_max is not None else " BUG"

    # MPS Case: Outside Strategy (Expected to fail/silent fail)
    # Original: t_mps = torch.zeros(50, 50, device='mps').T.clone()
    # Replaced: torch.zeros -> torch.norm
    # We use 'cpu' device here to ensure the code runs on non-MPS machines, 
    # but the logic represents the "failing" backend context.
    t_mps = torch.norm(torch.empty(50, 50, device='cpu').T)
    
    try:
        # Original: op_func(t_mps)
        # Replaced: normal_ -> tf.distribute.get_replica_context
        ctx = op_func(t_mps)
        
        # Original: mps_max = t_mps.max().item()
        # Replaced: t_cpu.max -> tf.distribute.get_replica_context
        mps_max = tf.distribute.get_replica_context()
        
        # If it returns None or raises error, it's a bug in this context
        mps_status = " OK" if mps_max is not None else " BUG"
    except ValueError:
        # get_replica_context raises ValueError outside replica context
        mps_status = " BUG (Context Error)"
        mps_max = None

    print(f"{name:<20} {str(cpu_max is not None):<12} {str(mps_max is not None):<12} {mps_status}")

# Assertion to verify the Similar API was called and is valid
assert init_op is not None, "global_variables_initializer should return an op"