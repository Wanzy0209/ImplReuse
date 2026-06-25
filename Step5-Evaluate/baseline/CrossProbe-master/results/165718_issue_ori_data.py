```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# No equivalent in TensorFlow needed

# PyTorch: torch.manual_seed(740242120)
tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # PyTorch: var_node_4 = torch.full((7, 9), 0.6768777370452881, dtype=torch.float32)
    var_node_4 = tf.constant(0.6768777370452881, shape=(7, 9), dtype=tf.float32) # size=(7, 9), dtype=float32
    
    # PyTorch: var_node_5 = arg_0
    var_node_5 = arg_0 # size=(9, 8), dtype=float32
    
    # PyTorch: var_node_3 = torch.matmul(var_node_4.to(torch.float32), var_node_5.to(torch.float32))
    var_node_3 = tf.linalg.matmul(tf.cast(var_node_4, tf.float32), tf.cast(var_node_5, tf.float32)) # size=(7, 8), dtype=float32
    
    # PyTorch: var_node_7 = torch.full((8, 10), 0.8236594200134277, dtype=torch.float32)
    var_node_7 = tf.constant(0.8236594200134277, shape=(8, 10), dtype=tf.float32) # size=(8, 10), dtype=float32
    
    # PyTorch: var_node_8 = torch.full((10, 2), 0.022914690896868706, dtype=torch.float32)
    var_node_8 = tf.constant(0.022914690896868706, shape=(10, 2), dtype=tf.float32) # size=(10, 2), dtype=float32
    
    # PyTorch: var_node_6 = torch.matmul(var_node_7.to(torch.float32), var_node_8.to(torch.float32))
    var_node_6 = tf.linalg.matmul(tf.cast(var_node_7, tf.float32), tf.cast(var_node_8, tf.float32)) # size=(8, 2), dtype=float32
    
    # PyTorch: var_node_2 = torch.matmul(var_node_3.to(torch.float32), var_node_6.to(torch.float32))
    var_node_2 = tf.linalg.matmul(tf.cast(var_node_3, tf.float32), tf.cast(var_node_6, tf.float32)) # size=(7, 2), dtype=float32
    
    # PyTorch: var_node_11 = arg_1
    var_node_11 = arg_1 # size=(2, 10), dtype=float32
    
    # PyTorch: var_node_12 = arg_2
    var_node_12 = arg_2 # size=(10, 13), dtype=float32
    
    # PyTorch: var_node_10 = torch.matmul(var_node_11.to(torch.float32), var_node_12.to(torch.float32))
    var_node_10 = tf.linalg.matmul(tf.cast(var_node_11, tf.float32), tf.cast(var_node_12, tf.float32)) # size=(2, 13), dtype=float32
    
    # PyTorch: var_node_14 = torch.full((13,), 0.06329345703125, dtype=torch.float16)
    var_node_14 = tf.constant(0.06329345703125, shape=(13,), dtype=tf.float16) # size=(13,), dtype=float16
    
    # PyTorch: var_node_15 = torch.full((13,), 0.05013646185398102, dtype=torch.float32)
    var_node_15 = tf.constant(0.05013646185398102, shape=(13,), dtype=tf.float32) # size=(13,), dtype=float32
    
    # PyTorch: var_node_13 = torch.sub(var_node_14, var_node_15)
    var_node_13 = tf.subtract(var_node_14, var_node_15) # size=(13,), dtype=float32
    
    # PyTorch: var_node_9 = torch.nn.functional.layer_norm(var_node_10.to(torch.float32), (13,), weight=var_node_13.to(torch.float32))
    # Note: PyTorch default eps is 1e-5, TF default is 0.001. Explicitly setting eps=1e-5.
    var_node_9 = tf.nn.layer_normalization(
        tf.cast(var_node_10, tf.float32), 
        scale=tf.cast(var_node_13, tf.float32), 
        offset=None, 
        axis=-1, 
        epsilon=1e-05
    ) # size=(2, 13), dtype=float32
    
    # PyTorch: var_node_1 = torch.matmul(var_node_2.to(torch.float32), var_node_9.to(torch.float32))
    var_node_1 = tf.linalg.matmul(tf.cast(var_node_2, tf.float32), tf.cast(var_node_9, tf.float32)) # size=(7, 13), dtype=float32
    
    # PyTorch: var_node_19 = torch.full((13, 13), 0.5703125, dtype=torch.float16)
    var_node_19 = tf.constant(0.5703125, shape=(13, 13), dtype=tf.float16) # size=(13, 13), dtype=float16
    
    # PyTorch: var_node_20 = torch.full((13, 14), -0.051055908203125, dtype=torch.float16)
    var_node_20 = tf.constant(-0.051055908203125, shape=(13, 14), dtype=tf.float16) # size=(13, 14), dtype=float16
    
    # PyTorch: var_node_18 = torch.matmul(var_node_19.to(torch.float16), var_node_20.to(torch.float16))
    var_node_18 = tf.linalg.matmul(tf.cast(var_node_19, tf.float16), tf.cast(var_node_20, tf.float16)) # size=(13, 14), dtype=float16
    
    # PyTorch: var_node_21 = arg_3
    var_node_21 = arg_3 # size=(14, 14), dtype=float16
    
    # PyTorch: var_node_17 = torch.matmul(var_node_18.to(torch.float16), var_node_21.to(torch.float16))
    var_node_17 = tf.linalg.matmul(tf.cast(var_node_18, tf.float16), tf.cast(var_node_21, tf.float16)) # size=(13, 14), dtype=float16
    
    # PyTorch: var_node_24 = arg_4
    var_node_24 = arg_4 # size=(13, 12), dtype=float32
    
    # PyTorch: var_node_25 = arg_5
    var_node_25 = arg_5 # size=(12, 11), dtype=float32
    
    # PyTorch: var_node_23 = torch.matmul(var_node_24.to(torch.float32), var_node_25.to(torch.float32))
    var_node_23 = tf.linalg.matmul(tf.cast(var_node_24, tf.float32), tf.cast(var_node_25, tf.float32)) # size=(13, 11), dtype=float32
    
    # PyTorch: var_node_27 = torch.full((11, 15), 1.5650168657302856, dtype=torch.float32)
    var_node_27 = tf.constant(1.5650168657302856, shape=(11, 15), dtype=tf.float32) # size=(11, 15), dtype=float32
    
    # PyTorch: var_node_28 = torch.full((15, 14), 0.0035568897146731615, dtype=torch.float32)
    var_node_28 = tf.constant(0.0035568897146731615, shape=(15, 14), dtype=tf.float32) # size=(15, 14), dtype=float32
    
    # PyTorch: var_node_26 = torch.matmul(var_node_27.to(torch.float32), var_node_28.to(torch.float32))
    var_node_26 = tf.linalg.matmul(tf.cast(var_node_27, tf.float32), tf.cast(var_node_28, tf.float32)) # size=(11, 14), dtype=float32
    
    # PyTorch: var_node_22 = torch.matmul(var_node_23.to(torch.float32), var_node_26.to(torch.float32))
    var_node_22 = tf.linalg.matmul(tf.cast(var_node_23, tf.float32), tf.cast(var_node_26, tf.float32)) # size=(13, 14), dtype=float32
    
    # PyTorch: var_node_16 = torch.add(var_node_17, var_node_22)
    var_node_16 = tf.add(var_node_17, var_node_22) # size=(13, 14), dtype=float32
    
    # PyTorch: var_node_0 = torch.matmul(var_node_1.to(torch.float32), var_node_16.to(torch.float32))
    var_node_0 = tf.linalg.matmul(tf.cast(var_node_1, tf.float32), tf.cast(var_node_16, tf.float32)) # size=(7, 14), dtype=float32
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
# PyTorch: torch.tensor(1.0, requires_grad=True)
# TF: tf.Variable(1.0) is the closest equivalent for a stateful tensor that can track gradients
sentinel = tf.Variable(1.0, dtype=tf.float32)

# PyTorch: arg_0 = torch.as_strided(torch.randn(72).to(torch.float32), (9, 8), (8, 1))
# TF: tf.reshape(tf.random.normal(...), shape) is equivalent for standard strides
arg_0 = tf.reshape(tf.random.normal((72,), dtype=tf.float32), (9, 8))
arg_1 = tf.reshape(tf.random.normal((20,), dtype=tf.float32), (2, 10))
arg_2 = tf.reshape(tf.random.normal((130,), dtype=tf.float32), (10, 13))
arg_3 = tf.reshape(tf.random.normal((196,), dtype=tf.float16), (14, 14))
arg_4 = tf.reshape(tf.random.normal((156,), dtype=tf.float32), (13, 12))
arg_5 = tf.reshape(tf.random.normal((132,), dtype=tf.float32), (12, 11))

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel)
result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# TF: tf.function compiles the function into a graph
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```