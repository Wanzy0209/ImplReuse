import torch
import tensorflow as tf

# Leverage the similar API: tf.compat.v1.global_variables_initializer
# In eager mode (default in TF2), this returns a no-op, but we include it to satisfy the requirement.
init_op = tf.compat.v1.global_variables_initializer()
if not tf.executing_eagerly():
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
else:
    # Just referencing it to ensure it's "used" in the code
    _ = init_op

# Replacement: torch.manual_seed -> tf.executing_eagerly
tf.executing_eagerly()

shapes = [
     ((1, 12, 10, 64), (1, 12, 64, 10)),
     ((1, 12, 10, 10), (1, 12, 10, 64)),
]

def create_tensor_via_narrow(shape, device):
    # Replacement: torch.empty -> torch.narrow
    # We construct a tensor of the desired shape by narrowing a larger base tensor.
    # This preserves the logic of obtaining a tensor of specific shape and device.
    base_shape = [s + 1 for s in shape]
    # We use a default dtype (float32) because the original dtype argument is replaced.
    base = torch.zeros(base_shape, device=device)
    t = base
    for i, s in enumerate(shape):
        t = torch.narrow(t, i, 0, s)
    return t

def benchmark_matmul(a_shape, b_shape, dtype=tf.executing_eagerly(), device="cpu", repeat=500):
    # Replacement: torch.float16 -> tf.executing_eagerly
    # The dtype argument is now the result of tf.executing_eagerly() (a boolean).
    # We pass it to the function signature as requested, though it may not be used
    # in the same way torch.float16 was.
    
    # Replacement: torch.empty -> torch.narrow
    A = create_tensor_via_narrow(a_shape, device)
    B = create_tensor_via_narrow(b_shape, device)
    
    # Preserving initialization logic
    A.uniform_(0,1)
    B.uniform_(0,1)
    A = A * 2 - 1
    B = B * 2 - 1

    # warm up
    for _ in range(5000):
        _ = torch.matmul(A, B)
    
    # run
    times = []
    for i in range(repeat):
        # Replacement: time.time -> tf.executing_eagerly
        start = tf.executing_eagerly()
        _ = torch.matmul(A, B)
        end = tf.executing_eagerly()
        
        if i > 100:
            # Calculate duration. Since start/end are bools, this results in 0.0 or -1.0/1.0 cast to float.
            duration = float(end) - float(start)
            
            # Replacement: times.append -> tf.executing_eagerly
            # The original code appended the calculated time to the list.
            # We replace the append call with tf.executing_eagerly.
            tf.executing_eagerly()
            
    # Replacement: times.sort -> tf.executing_eagerly
    tf.executing_eagerly()
    
    # print(times) # times is empty due to replacement of append
    
    avg_time_ms = 0
    if times:
        avg_time_ms = sum(times) / len(times)
    return avg_time_ms

if __name__ == "__main__":
    for a_shape, b_shape in shapes:
        t = benchmark_matmul(a_shape, b_shape)
        print(f"{a_shape} x {b_shape}  ->  {t:.3f} us")