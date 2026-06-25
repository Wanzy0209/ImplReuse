```python
import tensorflow as tf

# PyTorch: torch._dynamo.config.capture_scalar_outputs = True
# PyTorch: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Note: TensorFlow's tf.function handles graph capturing and retracing automatically.

# PyTorch: torch.manual_seed(52676)
tf.random.set_seed(52676)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    var_node_6 = arg_0 # size=(9, 9, 9), dtype=float64
    var_node_7 = arg_1 # size=(9, 9, 11), dtype=float64
    # PyTorch: torch.matmul
    var_node_5 = tf.matmul(tf.cast(var_node_6, tf.float64), tf.cast(var_node_7, tf.float64)) # size=(9, 9, 11), dtype=float64
    
    # PyTorch: torch.full
    var_node_9 = tf.constant(1.5758497316910556, shape=(9, 11, 12), dtype=tf.float64) # size=(9, 11, 12), dtype=float64
    
    var_node_10 = arg_2 # size=(9, 12, 8), dtype=float64
    var_node_8 = tf.matmul(tf.cast(var_node_9, tf.float64), tf.cast(var_node_10, tf.float64)) # size=(9, 11, 8), dtype=float64
    var_node_4 = tf.matmul(tf.cast(var_node_5, tf.float64), tf.cast(var_node_8, tf.float64)) # size=(9, 9, 8), dtype=float64
    
    var_node_13 = arg_3 # size=(9, 8, 13), dtype=float64
    var_node_14 = arg_4 # size=(9, 13, 7), dtype=float64
    var_node_12 = tf.matmul(tf.cast(var_node_13, tf.float64), tf.cast(var_node_14, tf.float64)) # size=(9, 8, 7), dtype=float64
    
    var_node_15 = arg_5 # size=(9, 7, 16), dtype=float64
    var_node_11 = tf.matmul(tf.cast(var_node_12, tf.float64), tf.cast(var_node_15, tf.float64)) # size=(9, 8, 16), dtype=float64
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.float64), tf.cast(var_node_11, tf.float64)) # size=(9, 9, 16), dtype=float64
    
    var_node_17 = arg_6 # size=(9, 16, 12), dtype=float64
    var_node_18 = arg_7 # size=(9, 12, 11), dtype=float64
    var_node_16 = tf.matmul(tf.cast(var_node_17, tf.float64), tf.cast(var_node_18, tf.float64)) # size=(9, 16, 11), dtype=float64
    var_node_2 = tf.matmul(tf.cast(var_node_3, tf.float64), tf.cast(var_node_16, tf.float64)) # size=(9, 9, 11), dtype=float64
    
    var_node_23 = tf.constant(-0.5249394453404403, shape=(156, 8), dtype=tf.float64) # size=(156, 8), dtype=float64
    var_node_24 = tf.constant(0.9331226188585692, shape=(8, 9), dtype=tf.float64) # size=(8, 9), dtype=float64
    var_node_22 = tf.matmul(tf.cast(var_node_23, tf.float64), tf.cast(var_node_24, tf.float64)) # size=(156, 9), dtype=float64
    
    var_node_26 = tf.constant(-0.9276381954691514, shape=(9, 13), dtype=tf.float64) # size=(9, 13), dtype=float64
    var_node_27 = tf.constant(0.024752238943232543, shape=(13, 16), dtype=tf.float64) # size=(13, 16), dtype=float64
    var_node_25 = tf.matmul(tf.cast(var_node_26, tf.float64), tf.cast(var_node_27, tf.float64)) # size=(9, 16), dtype=float64
    var_node_21 = tf.matmul(tf.cast(var_node_22, tf.float64), tf.cast(var_node_25, tf.float64)) # size=(156, 16), dtype=float64
    
    var_node_29 = arg_8 # size=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool
    
    # PyTorch: _x_nz = torch.zeros(...); _x_nz_flat[:9] = True; var_node_28 = torch.nonzero(_x_nz)
    # Since _x_nz has size 9, setting first 9 elements to True means all elements are True.
    # PyTorch: torch.nonzero
    _x_nz = tf.ones((9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=tf.bool)
    var_node_28 = tf.where(_x_nz) # size=(9, 11), dtype=int64
    
    # PyTorch: torch.nn.functional.embedding(torch.clamp(...), ...)
    # PyTorch: torch.clamp
    # TensorFlow: tf.nn.embedding_lookup, tf.clip_by_value
    var_node_28_clamped = tf.clip_by_value(tf.cast(var_node_28, tf.int64), 0, tf.shape(var_node_21)[0] - 1)
    var_node_20 = tf.nn.embedding_lookup(tf.cast(var_node_21, tf.float64), var_node_28_clamped) # size=(9, 11, 16), dtype=float64
    
    var_node_33 = tf.constant(1.0707914920634904, shape=(9, 16, 5), dtype=tf.float64) # size=(9, 16, 5), dtype=float64
    var_node_34 = tf.constant(-0.44934093079047227, shape=(9, 5, 10), dtype=tf.float64) # size=(9, 5, 10), dtype=float64
    var_node_32 = tf.matmul(tf.cast(var_node_33, tf.float64), tf.cast(var_node_34, tf.float64)) # size=(9, 16, 10), dtype=float64
    
    var_node_36 = arg_9 # size=(9, 10, 1), dtype=float64
    var_node_37 = tf.constant(-1.874293687140311, shape=(9, 1, 11), dtype=tf.float64) # size=(9, 1, 11), dtype=float64
    var_node_35 = tf.matmul(tf.cast(var_node_36, tf.float64), tf.cast(var_node_37, tf.float64)) # size=(9, 10, 11), dtype=float64
    var_node_31 = tf.matmul(tf.cast(var_node_32, tf.float64), tf.cast(var_node_35, tf.float64)) # size=(9, 16, 11), dtype=float64
    
    var_node_40 = tf.constant(0.4084376380351558, shape=(990, 2), dtype=tf.float64) # size=(990, 2), dtype=float64
    var_node_41 = tf.constant(0.982671965550022, shape=(2,), dtype=tf.float64) # size=(2,), dtype=float64
    var_node_39 = tf.matmul(tf.cast(var_node_40, tf.float64), tf.cast(var_node_41, tf.float64)) # size=(990,), dtype=float64
    
    # PyTorch: torch.reshape
    var_node_38 = tf.reshape(var_node_39, [9, 11, 10]) # size=(9, 11, 10), dtype=float64
    var_node_30 = tf.matmul(tf.cast(var_node_31, tf.float64), tf.cast(var_node_38, tf.float64)) # size=(9, 16, 10), dtype=float64
    var_node_19 = tf.matmul(tf.cast(var_node_20, tf.float64), tf.cast(var_node_30, tf.float64)) # size=(9, 11, 10), dtype=float64
    var_node_1 = tf.matmul(tf.cast(var_node_2, tf.float64), tf.cast(var_node_19, tf.float64)) # size=(9, 9, 10), dtype=float64
    
    var_node_47 = arg_10 # size=(9, 10, 15), dtype=float64
    var_node_48 = tf.constant(-0.3349339402390618, shape=(9, 15, 2), dtype=tf.float64) # size=(9, 15, 2), dtype=float64
    var_node_46 = tf.matmul(tf.cast(var_node_47, tf.float64), tf.cast(var_node_48, tf.float64)) # size=(9, 10, 2), dtype=float64
    
    var_node_50 = arg_11 # size=(9, 2, 7), dtype=float64
    var_node_51 = arg_12 # size=(9, 7, 2), dtype=float64
    var_node_49 = tf.matmul(tf.cast(var_node_50, tf.float64), tf.cast(var_node_51, tf.float64)) # size=(9, 2, 2), dtype=float64
    var_node_45 = tf.matmul(tf.cast(var_node_46, tf.float64), tf.cast(var_node_49, tf.float64)) # size=(9, 10, 2), dtype=float64
    
    var_node_52 = tf.constant(-0.4046675639434615, shape=(9, 2, 1), dtype=tf.float64) # size=(9, 2, 1), dtype=float64
    var_node_44 = tf.matmul(tf.cast(var_node_45, tf.float64), tf.cast(var_node_52, tf.float64)) # size=(9, 10, 1), dtype=float64
    
    var_node_56 = arg_13 # size=(9, 1, 1), dtype=float64
    # PyTorch: torch.nn.functional.rms_norm
    # TensorFlow: manual implementation of RMS Norm
    epsilon = 1e-5
    mean_sq = tf.reduce_mean(tf.math.square(tf.cast(var_node_56, tf.float64)), axis=-1, keepdims=True)
    var_node_55 = tf.cast(var_node_56, tf.float64) * tf.math.rsqrt(mean_sq + epsilon) # size=(9, 1, 1), dtype=float64
    
    var_node_57 = tf.constant(0.17877664640931384, shape=(9, 1, 8), dtype=tf.float64) # size=(9, 1, 8), dtype=float64
    var_node_54 = tf.matmul(tf.cast(var_node_55, tf.float64), tf.cast(var_node_57, tf.float64)) # size=(9, 1, 8), dtype=float64
    
    var_node_60 = arg_14 # size=(9, 8, 10), dtype=float64
    var_node_61 = tf.constant(0.43614806380221494, shape=(9, 10, 6), dtype=tf.float64) # size=(9, 10, 6), dtype=float64
    var_node_59 = tf.matmul(tf.cast(var_node_60, tf.float64), tf.cast(var_node_61, tf.float64)) # size=(9, 8, 6), dtype=float64
    
    var_node_63 = arg_15 # size=(9, 6, 3), dtype=float64
    var_node_64 = tf.constant(-0.042774422041922854, shape=(9, 3, 8), dtype=tf.float64) # size=(9, 3, 8), dtype=float64
    var_node_62 = tf.matmul(tf.cast(var_node_63, tf.float64), tf.cast(var_node_64, tf.float64)) # size=(9, 6, 8), dtype=float64
    var_node_58 = tf.matmul(tf.cast(var_node_59, tf.float64), tf.cast(var_node_62, tf.float64)) # size=(9, 8, 8), dtype=float64
    var_node_53 = tf.matmul(tf.cast(var_node_54, tf.float64), tf.cast(var_node_58, tf.float64)) # size=(9, 1, 8), dtype=float64
    var_node_43 = tf.matmul(tf.cast(var_node_44, tf.float64), tf.cast(var_node_53, tf.float64)) # size=(9, 10, 8), dtype=float64
    
    var_node_68 = arg_16 # size=(9, 8, 16), dtype=float64
    var_node_70 = tf.constant(0.24947808634496438, shape=(9, 16, 15), dtype=tf.float64) # size=(9, 16, 15), dtype=float64
    var_node_71 = tf.constant(-0.09035245509773453, shape=(9, 15, 7), dtype=tf.float64) # size=(9, 15, 7), dtype=float64
    var_node_69 = tf.matmul(tf.cast(var_node_70, tf.float64), tf.cast(var_node_71, tf.float64)) # size=(9, 16, 7), dtype=float64
    var_node_67 = tf.matmul(tf.cast(var_node_68, tf.float64), tf.cast(var_node_69, tf.float64)) # size=(9, 8, 7), dtype=float64
    
    var_node_74 = tf.constant(0.05671950481832341, shape=(9, 7, 1), dtype=tf.float64) # size=(9, 7, 1), dtype=float64
    # PyTorch: torch.nn.functional.gelu
    # TensorFlow: tf.nn.gelu
    var_node_73 = tf.nn.gelu(var_node_74) # size=(9, 7, 1), dtype=float64
    
    var_node_76 = tf.constant(-0.019912810353597852, shape=(9, 1, 2), dtype=tf.float64) # size=(9, 1, 2), dtype=float64
    var_node_77 = arg_17 # size=(9, 2, 7), dtype=float64
    var_node_75 = tf.matmul(tf.cast(var_node_76, tf.float64), tf.cast(var_node_77, tf.float64)) # size=(9, 1, 7), dtype=float64
    var_node_72 = tf.matmul(tf.cast(var_node_73, tf.float64), tf.cast(var_node_75, tf.float64)) # size=(9, 7, 7), dtype=float64
    var_node_66 = tf.matmul(tf.cast(var_node_67, tf.float64), tf.cast(var_node_72, tf.float64)) # size=(9, 8, 7), dtype=float64
    
    var_node_78 = arg_18 # size=(9, 7, 13), dtype=float64
    var_node_65 = tf.matmul(tf.cast(var_node_66, tf.float64), tf.cast(var_node_78, tf.float64)) # size=(9, 8, 13), dtype=float64
    var_node_42 = tf.matmul(tf.cast(var_node_43, tf.float64), tf.cast(var_node_65, tf.float64)) # size=(9, 10, 13), dtype=float64
    var_node_0 = tf.matmul(tf.cast(var_node_1, tf.float64), tf.cast(var_node_42, tf.float64)) # size=(9, 9, 13), dtype=float64
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if tf.math.is_complex(result):
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.constant(1.0, dtype=tf.float64)

# PyTorch: torch.as_strided(torch.randn(...).to(torch.float64), ...)
# TensorFlow: tf.random.normal (since strides are contiguous, as_strided is equivalent to reshape)
arg_0 = tf.random.normal((9, 9, 9), dtype=tf.float64)
arg_1 = tf.random.normal((9, 9, 11), dtype=tf.float64)
arg_2 = tf.random.normal((9, 12, 8), dtype=tf.float64)
arg_3 = tf.random.normal((9, 8, 13), dtype=tf.float64)
arg_4 = tf.random.normal((9, 13, 7), dtype=tf.float64)
arg_5 = tf.random.normal((9, 7, 16), dtype=tf.float64)
arg_6 = tf.random.normal((9, 16, 12), dtype=tf.float64)
arg_7 = tf.random.normal((9, 12, 11), dtype=tf.float64)

# PyTorch: torch.as_strided(torch.randint(0, 2, (1,), dtype=torch.int8).bool(), ...)
# TensorFlow: tf.cast(tf.random.uniform(...), tf.bool) then broadcast
arg_8_val = tf.cast(tf.random.uniform((), 0, 2, dtype=tf.int8), tf.bool)
arg_8 = tf.broadcast_to(arg_8_val, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

arg_9 = tf.random.normal((9, 10, 1), dtype=tf.float64)
arg_10 = tf.random.normal((9, 10, 15), dtype=tf.float64)
arg_11 = tf.random.normal((9, 2, 7), dtype=tf.float64)
arg_12 = tf.random.normal((9, 7, 2), dtype=tf.float64)
arg_13 = tf.random.normal((9, 1, 1), dtype=tf.float64)
arg_14 = tf.random.normal((9, 8, 10), dtype=tf.float64)
arg_15 = tf.random.normal((9, 6, 3), dtype=tf.float64)
arg_16 = tf.random.normal((9, 8, 16), dtype=tf.float64)
arg_17 = tf.random.normal((9, 2, 7), dtype=tf.float64)
arg_18 = tf.random.normal((9, 7, 13), dtype=tf.float64)

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel)

result_original = fuzzed_program(*args)
print('✅ eager success')

# PyTorch: torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# TensorFlow: tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```