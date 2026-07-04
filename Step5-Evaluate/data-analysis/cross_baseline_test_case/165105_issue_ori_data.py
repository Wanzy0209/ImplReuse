```python
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(70609)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel):
    # Conversion: torch.full -> tf.fill
    var_node_4 = tf.fill((14,), tf.cast(1.2255859375, tf.float16)) # size=(14,), dtype=float16
    var_node_6 = arg_0 # size=(14, 6), dtype=float16
    var_node_9 = tf.fill((6, 13), tf.cast(1.3154296875, tf.float16)) # size=(6, 13), dtype=float16
    var_node_10 = arg_1 # size=(13, 1), dtype=float16
    
    # Conversion: torch.matmul -> tf.matmul
    var_node_8 = tf.matmul(var_node_9, var_node_10) # size=(6, 1), dtype=float16
    var_node_12 = arg_2 # size=(1, 10), dtype=float16
    var_node_13 = tf.fill((10, 416), tf.cast(0.1331787109375, tf.float16)) # size=(10, 416), dtype=float16
    var_node_11 = tf.matmul(var_node_12, var_node_13) # size=(1, 416), dtype=float16
    var_node_7 = tf.matmul(var_node_8, var_node_11) # size=(6, 416), dtype=float16
    var_node_5 = tf.matmul(var_node_6, var_node_7) # size=(14, 416), dtype=float16
    var_node_3 = tf.matmul(var_node_4, var_node_5) # size=(416,), dtype=float16
    
    # Conversion: .view() -> tf.reshape
    var_node_2 = tf.reshape(var_node_3, [26, 1, 16]) # size=(26, 1, 16), dtype=float16
    
    var_node_16 = arg_3 # size=(329, 4), dtype=float16
    var_node_20 = arg_4 # size=(1,), dtype=bool
    var_node_19 = tf.reshape(var_node_20, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) # size=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool
    
    # Conversion: torch.unsqueeze -> tf.expand_dims
    var_node_18 = tf.expand_dims(var_node_19, axis=15) # size=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=bool
    
    # Conversion: torch.zeros + slice assignment + torch.nonzero -> tf.ones + tf.where
    # The logic creates a mask of all True values for the first 26 elements in a flattened view of shape (26, 1...1).
    # Since the total size is 26, the mask is all True.
    _x_nz = tf.ones((26, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=tf.bool)
    var_node_17 = tf.where(_x_nz) # size=(26, 16), dtype=int64
    
    # Conversion: torch.clamp -> tf.clip_by_value, torch.nn.functional.embedding -> tf.nn.embedding_lookup
    # Note: embedding_lookup expects int32 indices
    indices = tf.cast(tf.clip_by_value(var_node_17, 0, tf.shape(var_node_16)[0] - 1), tf.int32)
    var_node_15 = tf.nn.embedding_lookup(var_node_16, indices) # size=(26, 16, 4), dtype=float16
    
    var_node_25 = arg_5 # size=(26, 4, 8), dtype=float16
    var_node_26 = arg_6 # size=(26, 8, 11), dtype=float16
    var_node_24 = tf.matmul(var_node_25, var_node_26) # size=(26, 4, 11), dtype=float16
    var_node_28 = tf.fill((26, 11, 16), tf.cast(-0.73291015625, tf.float16)) # size=(26, 11, 16), dtype=float16
    var_node_29 = tf.fill((26, 16, 16), tf.cast(-0.53271484375, tf.float16)) # size=(26, 16, 16), dtype=float16
    var_node_27 = tf.matmul(var_node_28, var_node_29) # size=(26, 11, 16), dtype=float16
    var_node_23 = tf.matmul(var_node_24, var_node_27) # size=(26, 4, 16), dtype=float16
    var_node_31 = arg_7 # size=(26, 16, 11), dtype=float16
    var_node_33 = arg_8 # size=(26, 11, 11), dtype=float16
    var_node_34 = tf.fill((26, 11, 6), tf.cast(-0.51318359375, tf.float16)) # size=(26, 11, 6), dtype=float16
    var_node_32 = tf.matmul(var_node_33, var_node_34) # size=(26, 11, 6), dtype=float16
    var_node_30 = tf.matmul(var_node_31, var_node_32) # size=(26, 16, 6), dtype=float16
    var_node_22 = tf.matmul(var_node_23, var_node_30) # size=(26, 4, 6), dtype=float16
    var_node_38 = tf.fill((26, 6, 3), tf.cast(0.56591796875, tf.float16)) # size=(26, 6, 3), dtype=float16
    var_node_39 = arg_9 # size=(26, 3, 11), dtype=float16
    var_node_37 = tf.matmul(var_node_38, var_node_39) # size=(26, 6, 11), dtype=float16
    var_node_41 = arg_10 # size=(26, 11, 6), dtype=float16
    var_node_42 = tf.fill((26, 6, 3), tf.cast(-0.69580078125, tf.float16)) # size=(26, 6, 3), dtype=float16
    var_node_40 = tf.matmul(var_node_41, var_node_42) # size=(26, 11, 3), dtype=float16
    var_node_36 = tf.matmul(var_node_37, var_node_40) # size=(26, 6, 3), dtype=float16
    var_node_45 = arg_11 # size=(26, 3, 11), dtype=float16
    var_node_46 = arg_12 # size=(26, 11, 15), dtype=float16
    var_node_44 = tf.matmul(var_node_45, var_node_46) # size=(26, 3, 15), dtype=float16
    var_node_48 = tf.fill((26, 15, 2), tf.cast(-1.2646484375, tf.float16)) # size=(26, 15, 2), dtype=float16
    var_node_49 = arg_13 # size=(26, 2, 5), dtype=float16
    var_node_47 = tf.matmul(var_node_48, var_node_49) # size=(26, 15, 5), dtype=float16
    var_node_43 = tf.matmul(var_node_44, var_node_47) # size=(26, 3, 5), dtype=float16
    var_node_35 = tf.matmul(var_node_36, var_node_43) # size=(26, 6, 5), dtype=float16
    var_node_21 = tf.matmul(var_node_22, var_node_35) # size=(26, 4, 5), dtype=float16
    var_node_14 = tf.matmul(var_node_15, var_node_21) # size=(26, 16, 5), dtype=float16
    var_node_1 = tf.matmul(var_node_2, var_node_14) # size=(26, 1, 5), dtype=float16
    var_node_56 = tf.fill((1, 26, 5, 5), tf.cast(-2.099609375, tf.float16)) # size=(1, 26, 5, 5), dtype=float16
    var_node_57 = tf.fill((1, 26, 5, 10), tf.cast(-1.5166015625, tf.float16)) # size=(1, 26, 5, 10), dtype=float16
    var_node_55 = tf.matmul(var_node_56, var_node_57) # size=(1, 26, 5, 10), dtype=float16
    
    # Conversion: torch.squeeze -> tf.squeeze
    var_node_54 = tf.squeeze(var_node_55) # size=(26, 5, 10), dtype=float16
    
    var_node_60 = arg_14 # size=(26, 10, 9), dtype=float16
    var_node_61 = tf.fill((26, 9, 13), tf.cast(-0.97509765625, tf.float16)) # size=(26, 9, 13), dtype=float16
    var_node_59 = tf.matmul(var_node_60, var_node_61) # size=(26, 10, 13), dtype=float16
    var_node_63 = tf.fill((26, 13, 5), tf.cast(0.77294921875, tf.float16)) # size=(26, 13, 5), dtype=float16
    var_node_64 = tf.fill((26, 5, 10), tf.cast(-1.0751953125, tf.float16)) # size=(26, 5, 10), dtype=float16
    var_node_62 = tf.matmul(var_node_63, var_node_64) # size=(26, 13, 10), dtype=float16
    var_node_58 = tf.matmul(var_node_59, var_node_62) # size=(26, 10, 10), dtype=float16
    var_node_53 = tf.matmul(var_node_54, var_node_58) # size=(26, 5, 10), dtype=float16
    var_node_68 = tf.fill((26, 10, 7), tf.cast(0.25439453125, tf.float16)) # size=(26, 10, 7), dtype=float16
    var_node_69 = arg_15 # size=(26, 7, 1), dtype=float16
    var_node_67 = tf.matmul(var_node_68, var_node_69) # size=(26, 10, 1), dtype=float16
    var_node_71 = arg_16 # size=(26, 1, 10), dtype=float16
    
    # Conversion: torch.nn.functional.silu -> tf.nn.silu
    var_node_70 = tf.nn.silu(var_node_71) # size=(26, 1, 10), dtype=float16
    
    var_node_66 = tf.matmul(var_node_67, var_node_70) # size=(26, 10, 10), dtype=float16
    var_node_72 = arg_17 # size=(26, 10, 10), dtype=float16
    var_node_65 = tf.matmul(var_node_66, var_node_72) # size=(26, 10, 10), dtype=float16
    var_node_52 = tf.matmul(var_node_53, var_node_65) # size=(26, 5, 10), dtype=float16
    var_node_76 = tf.fill((26, 10, 12), tf.cast(0.461669921875, tf.float16)) # size=(26, 10, 12), dtype=float16
    var_node_77 = tf.fill((26, 12, 5), tf.cast(0.79833984375, tf.float16)) # size=(26, 12, 5), dtype=float16
    var_node_75 = tf.matmul(var_node_76, var_node_77) # size=(26, 10, 5), dtype=float16
    var_node_80 = arg_18 # size=(26, 5, 5), dtype=float16
    var_node_81 = arg_19 # size=(26, 5, 6), dtype=float16
    var_node_79 = tf.matmul(var_node_80, var_node_81) # size=(26, 5, 6), dtype=float16
    var_node_83 = tf.fill((26, 6, 3), tf.cast(0.98974609375, tf.float16)) # size=(26, 6, 3), dtype=float16
    var_node_84 = arg_20 # size=(26, 3, 2), dtype=float16
    var_node_82 = tf.matmul(var_node_83, var_node_84) # size=(26, 6, 2), dtype=float16
    var_node_78 = tf.matmul(var_node_79, var_node_82) # size=(26, 5, 2), dtype=float16
    var_node_74 = tf.matmul(var_node_75, var_node_78) # size=(26, 10, 2), dtype=float16
    var_node_88 = tf.fill((26, 2, 1, 9), tf.cast(1.5703125, tf.float16)) # size=(26, 2, 1, 9), dtype=float16
    var_node_87 = tf.squeeze(var_node_88) # size=(26, 2, 9), dtype=float16
    var_node_90 = tf.fill((26, 9, 5), tf.cast(-1.150390625, tf.float16)) # size=(26, 9, 5), dtype=float16
    var_node_91 = tf.fill((26, 5, 14), tf.cast(-0.359619140625, tf.float16)) # size=(26, 5, 14), dtype=float16
    var_node_89 = tf.matmul(var_node_90, var_node_91) # size=(26, 9, 14), dtype=float16
    var_node_86 = tf.matmul(var_node_87, var_node_89) # size=(26, 2, 14), dtype=float16
    var_node_94 = arg_21 # size=(26, 14, 10), dtype=float16
    var_node_95 = tf.fill((26, 10, 7), tf.cast(-0.229248046875, tf.float16)) # size=(26, 10, 7), dtype=float16
    var_node_93 = tf.matmul(var_node_94, var_node_95) # size=(26, 14, 7), dtype=float16
    var_node_97 = arg_22 # size=(26, 7, 15), dtype=float16
    var_node_98 = tf.fill((26, 15, 4), tf.cast(-0.71533203125, tf.float16)) # size=(26, 15, 4), dtype=float16
    var_node_96 = tf.matmul(var_node_97, var_node_98) # size=(26, 7, 4), dtype=float16
    var_node_92 = tf.matmul(var_node_93, var_node_96) # size=(26, 14, 4), dtype=float16
    var_node_85 = tf.matmul(var_node_86, var_node_92) # size=(26, 2, 4), dtype=float16
    var_node_73 = tf.matmul(var_node_74, var_node_85) # size=(26, 10, 4), dtype=float16
    var_node_51 = tf.matmul(var_node_52, var_node_73) # size=(26, 5, 4), dtype=float16
    var_node_104 = arg_23 # size=(26, 4, 2), dtype=float16
    var_node_105 = tf.fill((26, 2, 9), tf.cast(-2.302734375, tf.float16)) # size=(26, 2, 9), dtype=float16
    var_node_103 = tf.matmul(var_node_104, var_node_105) # size=(26, 4, 9), dtype=float16
    var_node_107 = tf.fill((26, 9, 5), tf.cast(1.0341796875, tf.float16)) # size=(26, 9, 5), dtype=float16
    var_node_108 = arg_24 # size=(26, 5, 13), dtype=float16
    var_node_106 = tf.matmul(var_node_107, var_node_108) # size=(26, 9, 13), dtype=float16
    var_node_102 = tf.matmul(var_node_103, var_node_106) # size=(26, 4, 13), dtype=float16
    var_node_111 = tf.fill((26, 13, 3), tf.cast(-1.236328125, tf.float16)) # size=(26, 13, 3), dtype=float16
    var_node_112 = arg_25 # size=(26, 3, 8), dtype=float16
    var_node_110 = tf.matmul(var_node_111, var_node_112) # size=(26, 13, 8), dtype=float16
    var_node_114 = tf.fill((9,), tf.cast(1.4638671875, tf.float16)) # size=(9,), dtype=float16
    var_node_115 = tf.fill((9, 13), tf.cast(0.19775390625, tf.float16)) # size=(9, 13), dtype=float16
    var_node_113 = tf.matmul(var_node_114, var_node_115) # size=(13,), dtype=float16
    var_node_117 = tf.fill((7,), tf.cast(-0.64501953125, tf.float16)) # size=(7,), dtype=float16
    var_node_118 = arg_26 # size=(7, 13), dtype=float16
    var_node_116 = tf.matmul(var_node_117, var_node_118) # size=(13,), dtype=float16
    
    # Conversion: torch.nn.functional.batch_norm -> tf.nn.batch_normalization
    # Note: PyTorch default eps is 1e-5. training=False means using provided mean/var.
    var_node_109 = tf.nn.batch_normalization(var_node_110, var_node_113, var_node_116, offset=None, scale=None, variance_epsilon=1e-5) # size=(26, 13, 8), dtype=float16
    
    var_node_101 = tf.matmul(var_node_102, var_node_109) # size=(26, 4, 8), dtype=float16
    var_node_120 = tf.fill((26, 8), tf.cast(0.8681640625, tf.float16)) # size=(26, 8), dtype=float16
    var_node_119 = tf.expand_dims(var_node_120, axis=2) # size=(26, 8, 1), dtype=float16
    var_node_100 = tf.matmul(var_node_101, var_node_119) # size=(26, 4, 1), dtype=float16
    
    # Conversion: torch.nn.functional.dropout with training=False -> Identity
    var_node_99 = var_node_100 # size=(26, 4, 1), dtype=float16
    
    var_node_50 = tf.matmul(var_node_51, var_node_99) # size=(26, 5, 1), dtype=float16
    var_node_0 = tf.matmul(var_node_1, var_node_50) # size=(26, 1, 1), dtype=float16
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.dtype.is_complex:
        result = tf.math.real(result)
    return result

# Sentinel tensor to ensure gradient computation
sentinel = tf.Variable(1.0, dtype=tf.float16)

# Conversion: torch.as_strided + torch.randn/randint -> tf.reshape + tf.random
arg_0 = tf.reshape(tf.random.normal((84,), dtype=tf.float16), (14, 6))
arg_1 = tf.reshape(tf.random.normal((13,), dtype=tf.float16), (13, 1))
arg_2 = tf.reshape(tf.random.normal((10,), dtype=tf.float16), (1, 10))
arg_3 = tf.reshape(tf.random.normal((1316,), dtype=tf.float16), (329, 4))
arg_4 = tf.cast(tf.random.uniform((1,), 0, 2, dtype=tf.int32), tf.bool)
arg_5 = tf.reshape(tf.random.normal((832,), dtype=tf.float16), (26, 4, 8))
arg_6 = tf.reshape(tf.random.normal((2288,), dtype=tf.float16), (26, 8, 11))
arg_7 = tf.reshape(tf.random.normal((4576,), dtype=tf.float16), (26, 16, 11))
arg_8 = tf.reshape(tf.random.normal((3146,), dtype=tf.float16), (26, 11, 11))
arg_9 = tf.reshape(tf.random.normal((858,), dtype=tf.float16), (26, 3, 11))
arg_10 = tf.reshape(tf.random.normal((1716,), dtype=tf.float16), (26, 11, 6))
arg_11 = tf.reshape(tf.random.normal((858,), dtype=tf.float16), (26, 3, 11))
arg_12 = tf.reshape(tf.random.normal((4290,), dtype=tf.float16), (26, 11, 15))
arg_13 = tf.reshape(tf.random.normal((260,), dtype=tf.float16), (26, 2, 5))
arg_14 = tf.reshape(tf.random.normal((2340,), dtype=tf.float16), (26, 10, 9))
arg_15 = tf.reshape(tf.random.normal((182,), dtype=tf.float16), (26, 7, 1))
arg_16 = tf.reshape(tf.random.normal((260,), dtype=tf.float16), (26, 1, 10))
arg_17 = tf.reshape(tf.random.normal((2600,), dtype=tf.float16), (26, 10, 10))
arg_18 = tf.reshape(tf.random.normal((650,), dtype=tf.float16), (26, 5, 5))
arg_19 = tf.reshape(tf.random.normal((780,), dtype=tf.float16), (26, 5, 6))
arg_20 = tf.reshape(tf.random.normal((156,), dtype=tf.float16), (26, 3, 2))
arg_21 = tf.reshape(tf.random.normal((3640,), dtype=tf.float16), (26, 14, 10))
arg_22 = tf.reshape(tf.random.normal((2730,), dtype=tf.float16), (26, 7, 15))
arg_23 = tf.reshape(tf.random.normal((208,), dtype=tf.float16), (26, 4, 2))
arg_24 = tf.reshape(tf.random.normal((1690,), dtype=tf.float16), (26, 5, 13))
arg_25 = tf.reshape(tf.random.normal((624,), dtype=tf.float16), (26, 3, 8))
arg_26 = tf.reshape(tf.random.normal((91,), dtype=tf.float16), (7, 13))

args = (arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, arg_19, arg_20, arg_21, arg_22, arg_23, arg_24, arg_25, arg_26, sentinel)

result_original = fuzzed_program(*args)
print('✅ eager success')

# Conversion: torch.compile -> tf.function
compiled_program = tf.function(fuzzed_program)
result_compiled = compiled_program(*args)
print('✅ compile success')
```