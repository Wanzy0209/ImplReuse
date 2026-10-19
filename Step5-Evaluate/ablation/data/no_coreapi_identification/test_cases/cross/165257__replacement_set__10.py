import torch
import tensorflow as tf

# Define the replacement operations as tf.functions (which are instances of tf.types.experimental.PolymorphicFunction)
# Replacement: T.clone -> tf.types.experimental.PolymorphicFunction
@tf.function
def clone_op(tensor):
    return tf.identity(tensor)

# Replacement: normal_, uniform_, exponential_, random_, bernoulli_ -> tf.types.experimental.PolymorphicFunction
# We use a generic random operation to simulate the in-place behavior
@tf.function
def random_op(var):
    var.assign(tf.random.normal(var.shape))

# Replacement: t_cpu.max -> tf.types.experimental.PolymorphicFunction
@tf.function
def max_op(var):
    return tf.reduce_max(var)

# Test each operation on CPU and MPS
ops = [
    ("normal_(0,1)", lambda t: random_op(t)),
    ("uniform_(0,1)", lambda t: random_op(t)),
    ("exponential_(1)", lambda t: random_op(t)),
    ("bernoulli_(0.5)", lambda t: random_op(t)),
    ("random_()", lambda t: random_op(t)),
    ("random_(10)", lambda t: random_op(t)),
    ("random_(0,10)", lambda t: random_op(t)),
]

print(f"{'Operation':<20} {'CPU Max':<12} {'MPS Max':<12} {'Status'}")
print("-" * 60)

for name, op_func in ops:
    # CPU: works correctly
    # Original: t_cpu = torch.zeros(50, 50, device='cpu').T.clone()
    # Replacement: torch.zeros -> torch.qr
    # Replacement: T.clone -> tf.types.experimental.PolymorphicFunction (clone_op)
    
    # Create tensor using torch.qr
    input_mat = torch.randn(50, 50, device='cpu')
    q, _ = torch.qr(input_mat)
    t_torch = q.T
    
    # Convert to TF Variable
    t_cpu = tf.Variable(t_torch.numpy())
    
    # Leverage similar API: tf.compat.v1.global_variables_initializer
    # In TF2 eager mode, variables are initialized on creation, but we include the call
    # to reflect the relationship to the similar API.
    init_op = tf.compat.v1.global_variables_initializer()
    # Note: In eager mode, init_op is a no-op, but we acknowledge the API usage.

    # Apply clone replacement
    t_cpu = clone_op(t_cpu)
    t_cpu = tf.Variable(t_cpu)

    # Apply random op replacement
    op_func(t_cpu)

    # Apply max replacement
    cpu_max = max_op(t_cpu).numpy()

    # MPS: fails silently
    # Original: t_mps = torch.zeros(50, 50, device='mps').T.clone()
    # Replacement: torch.zeros -> torch.qr
    
    mps_max = 0.0
    try:
        # Create tensor using torch.qr on MPS
        input_mat_mps = torch.randn(50, 50, device='mps')
        q_mps, _ = torch.qr(input_mat_mps)
        t_torch_mps = q_mps.T
        
        # Convert to TF Variable (must move to CPU for TF compatibility)
        t_mps = tf.Variable(t_torch_mps.cpu().numpy())
        
        # Initialize
        # (Same as CPU)

        # Apply clone replacement
        t_mps = clone_op(t_mps)
        t_mps = tf.Variable(t_mps)

        # Apply random op replacement
        op_func(t_mps)

        # Apply max replacement
        mps_max = max_op(t_mps).numpy()
    except Exception as e:
        # Handle cases where MPS might not be available or fails
        print(f"Error in MPS path: {e}")
        mps_max = 0.0

    status = " OK" if mps_max != 0.0 else " BUG"
    print(f"{name:<20} {cpu_max:<12.4f} {mps_max:<12.4f} {status}")