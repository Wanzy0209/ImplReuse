import tensorflow as tf

# Define a set of polymorphic functions (tf.function) to test.
# These mimic the random operations in the original bug report.
# tf.function returns a PolymorphicFunction.
@tf.function
def uniform_op(t):
    return tf.random.uniform(tf.shape(t), 0, 1)

@tf.function
def normal_op(t):
    return tf.random.normal(tf.shape(t), 0, 1)

@tf.function
def poisson_op(t):
    return tf.random.poisson(tf.shape(t), 1.0)

ops = [
    ("uniform_op", uniform_op),
    ("normal_op", normal_op),
    ("poisson_op", poisson_op),
]

# Define inputs with different properties (Standard vs Transposed).
# This mimics the CPU vs MPS / Contiguous vs Non-contiguous comparison
# in the original bug report to test how the PolymorphicFunction handles
# different tensor shapes/layouts.
inputs = [
    ("Standard", tf.zeros((50, 50), dtype=tf.float32)),
    # Fixed: Use tf.transpose() instead of .transpose()
    ("Transposed", tf.transpose(tf.zeros((50, 50), dtype=tf.float32)))
]

print(f"{'Op':<15} {'Input':<15} {'Max':<10} {'Status'}")
print("-" * 50)

for op_name, op_func in ops:
    for input_name, input_tensor in inputs:
        # Use get_concrete_function to verify specialization based on input type.
        # This leverages the specific API: tf.types.experimental.PolymorphicFunction
        try:
            concrete = op_func.get_concrete_function(input_tensor)
            result = concrete(input_tensor)
            max_val = tf.reduce_max(result).numpy()
            
            # Check if operation produced non-zero results (mimicking the original bug check)
            status = " OK" if max_val != 0.0 else " BUG"
            print(f"{op_name:<15} {input_name:<15} {max_val:<10.4f} {status}")
        except Exception as e:
            print(f"{op_name:<15} {input_name:<15} {'ERROR':<10}  {e}")