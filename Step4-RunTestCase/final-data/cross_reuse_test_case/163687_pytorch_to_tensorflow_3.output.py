import tensorflow as tf

# Enable eager execution as per the API under test.
# This corresponds to the "eager" mode in the PyTorch bug report where the code works.
tf.compat.v1.enable_eager_execution()

def flex_attention(q, k, v):
    """
    TensorFlow implementation of a scaled dot-product attention mechanism
    to mimic PyTorch's flex_attention behavior.
    Inputs:
        q: Query tensor of shape (Batch, Heads, SeqLen_Q, Dim)
        k: Key tensor of shape (Batch, Heads, SeqLen_K, Dim)
        v: Value tensor of shape (Batch, Heads, SeqLen_K, Dim)
    Output:
        Tensor of shape (Batch, Heads, SeqLen_Q, Dim)
    """
    # Matmul of Q and K transpose
    # Explicitly transpose k to avoid potential cuBLAS issues with transpose_b=True
    # on incompatible hardware/driver combinations (e.g., RTX 4090 with CUDA 10.0)
    k_transposed = tf.transpose(k, [0, 1, 3, 2])
    matmul_qk = tf.matmul(q, k_transposed)
    
    # Scale logits
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Softmax normalization
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # Matmul with V
    output = tf.matmul(attention_weights, v)
    return output

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    # Replicating the tensor operations from the PyTorch bug report
    t0 = arg0
    t1 = arg1
    t2 = arg2
    t3 = flex_attention(t0, t1, t2)

    t4 = arg3
    t5 = arg4
    t6 = arg5
    t7 = flex_attention(t4, t5, t6)

    t8 = flex_attention(t3, t7, t7)

    t9 = arg6
    t10 = arg7
    t11 = flex_attention(t9, t7, t10)

    t12 = flex_attention(t11, t8, t3)

    t13 = arg8
    t14 = arg9
    t15 = flex_attention(t13, t2, t14)

    t16 = arg10
    # PyTorch: t17 = t16.clone(); t17.zero_()
    # TensorFlow equivalent: create a tensor of zeros with the same shape and dtype
    t17 = tf.zeros_like(t16)

    t18 = flex_attention(t17, t8, t3)

    t19 = flex_attention(t15, t17, t18)

    t20 = flex_attention(t8, t12, t19)

    output = t20
    return output

# Initialize inputs with shapes matching the PyTorch bug report
# PyTorch device 'cuda' is mapped to the default TensorFlow device (GPU if available, else CPU)
arg0 = tf.random.normal([27, 26, 62, 122])
arg1 = tf.random.normal([27, 26, 124, 122])
arg2 = tf.random.normal([27, 26, 124, 122])
arg3 = tf.random.normal([27, 26, 124, 122])
arg4 = tf.random.normal([27, 26, 248, 122])
arg5 = tf.random.normal([27, 26, 248, 122])
arg6 = tf.random.normal([27, 26, 31, 122])
arg7 = tf.random.normal([27, 26, 124, 122])
arg8 = tf.random.normal([27, 26, 31, 122])
arg9 = tf.random.normal([27, 26, 124, 122])
arg10 = tf.random.normal([27, 26, 124, 122])

# Execute the function
print("Executing test case with tf.compat.v1.enable_eager_execution...")
result = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# Verify the output shape matches the expected shape from the PyTorch issue
# Expected shape: (27, 26, 62, 122)
expected_shape = (27, 26, 62, 122)
assert result.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {result.shape}"

print("Test passed successfully.")