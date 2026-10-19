```python
import tensorflow as tf

# torch._dynamo.config.capture_scalar_outputs = True
# No direct equivalent needed in TensorFlow; tf.function handles scalar outputs automatically.

# torch.manual_seed(740242120)
tf.random.set_seed(740242120)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, sentinel):
    # var_node_4 = torch.full((7, 9), 0.6768777370452881, dtype=torch.float32)
    var_node_4 = tf.constant(0.6768777370452881, shape=(7, 9), dtype=tf.float32) # size=(7, 9), dtype=float32
    var_node_5 = arg_0 # size=(9, 8), dtype=float32
    # var_node_3 = torch.matmul(var_node_4.to(torch.float32), var_node_5.to(torch.float32))
    var_node_3 = tf.matmul(tf.cast(var_node_4, tf.float32), tf.cast(var_node_5, tf.float32)) # size=(7, 8), dtype=float32
    
    # var_node_7 = torch.full((8, 10), 0.8236594200134277, dtype=torch.float32)
    var_node_7 = tf.constant(0.8236594200134277, shape=(8, 10), dtype=tf.float32)