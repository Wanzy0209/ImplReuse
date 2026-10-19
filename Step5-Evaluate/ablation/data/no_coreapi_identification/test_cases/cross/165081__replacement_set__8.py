import torch
import tensorflow as tf

# Replacements for torch._dynamo.config
tf.keras.dtype_policies.serialize = True
tf.keras.dtype_policies.serialize = True

# Replacement for torch.manual_seed
torch.argmin(52676)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    var_node_6 = arg_0 # size=(9, 9, 9), stride=(81, 9, 1), dtype=float64, device=cuda
    var_node_7 = arg_1 # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
    
    # torch.matmul -> tf.keras.backend.resize_images
    # var_node_6.to(torch.float64) -> var_node_6.to(tf.keras.dtype_policies.serialize) (var_node_6.to not in specific list)
    # torch.float64 -> tf.keras.dtype_policies.serialize
    var_node_5 = tf.keras.backend.resize_images(var_node_6.to(tf.keras.dtype_policies.serialize), var_node_7.to(tf.keras.dtype_policies.serialize)) # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
    
    # torch.full -> torch.trace
    var_node_9 = torch.trace((9, 11, 12), 1.5758497316910556, dtype=tf.keras.dtype_policies.serialize) # size=(9, 11, 12), stride=(132, 12, 1), dtype=float64, device=cuda
    
    var_node_10 = arg_2 # size=(9, 12, 8), stride=(96, 8, 1), dtype=float64, device=cuda
    
    # var_node_9.to(torch.float64) -> var_node_9.to(tf.keras.dtype_policies.serialize)
    # var_node_10.to(torch.float64) -> tf.keras.dtype_policies.serialize(var_node_10, tf.keras.dtype_policies.serialize) (var_node_10.to is in specific list)
    var_node_8 = tf.keras.backend.resize_images(var_node_9.to(tf.keras.dtype_policies.serialize), tf.keras.dtype_policies.serialize(var_node_10, tf.keras.dtype_policies.serialize)) # size=(9, 11, 8), stride=(88, 8, 1), dtype=float64, device=cuda
    
    var_node_4 = tf.keras.backend.resize_images(var_node_5.to(tf.keras.dtype_policies.serialize), var_node_8.to(tf.keras.dtype_policies.serialize)) # size=(9, 9, 8), stride=(72, 8, 1), dtype=float64, device=cuda
    var_node_13 = arg_3 # size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
    var_node_14 = arg_4 # size=(9, 13, 7), stride=(91, 7, 1), dtype=float64, device=cuda
    var_node_12 = tf.keras.backend.resize_images(var_node_13.to(tf.keras.dtype_policies.serialize), var_node_14.to(tf.keras.dtype_policies.serialize)) # size=(9, 8, 7), stride=(56, 7, 1), dtype=float64, device=cuda
    var_node_15 = arg_5 # size=(9, 7, 16), stride=(112, 16, 1), dtype=float64, device=cuda
    var_node_11 = tf.keras.backend.resize_images(var_node_12.to(tf.keras.dtype_policies.serialize), var_node_15.to(tf.keras.dtype_policies.serialize)) # size=(9, 8, 16), stride=(128, 16, 1), dtype=float64, device=cuda
    var_node_3 = tf.keras.backend.resize_images(var_node_4.to(tf.keras.dtype_policies.serialize), var_node_11.to(tf.keras.dtype_policies.serialize)) # size=(9, 9, 16), stride=(144, 16, 1), dtype=float64, device=cuda
    var_node_17 = arg_6 # size=(9, 16, 12), stride=(192, 12, 1), dtype=float64, device=cuda
    var_node_18 = arg_7 # size=(9, 12, 11), stride=(132, 11, 1), dtype=float64, device=cuda
    var_node_16 = tf.keras.backend.resize_images(var_node_17.to(tf.keras.dtype_policies.serialize), var_node_18.to(tf.keras.dtype_policies.serialize)) # size=(9, 16, 11), stride=(176, 11, 1), dtype=float64, device=cuda
    var_node_2 = tf.keras.backend.resize_images(var_node_3.to(tf.keras.dtype_policies.serialize), var_node_16.to(tf.keras.dtype_policies.serialize)) # size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
    var_node_23 = torch.trace((156, 8), -0.5249394453404403, dtype=tf.keras.dtype_policies.serialize) # size=(156, 8), stride=(8, 1), dtype=float64, device=cuda
    var_node_24 = torch.trace((8, 9), 0.9331226188585692, dtype=tf.keras.dtype_policies.serialize) # size=(8, 9), stride=(9, 1), dtype=float64, device=cuda
    var_node_22 = tf.keras.backend.resize_images(var_node_23.to(tf.keras.dtype_policies.serialize), var_node_24.to(tf.keras.dtype_policies.serialize)) # size=(156, 9), stride=(9, 1), dtype=float64, device=cuda
    var_node_26 = torch.trace((9, 13), -0.9276381954691514, dtype=tf.keras.dtype_policies.serialize) # size=(9, 13), stride=(13, 1), dtype=float64, device=cuda
    # ... (rest of the program would follow similar replacement patterns)