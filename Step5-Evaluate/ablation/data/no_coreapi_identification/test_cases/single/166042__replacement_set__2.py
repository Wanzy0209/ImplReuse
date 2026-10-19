import torch
import tensorflow as tf

# Replacement for torch._dynamo.config.capture_scalar_outputs = True
tf.keras.backend.backend()

# Replacement for torch.manual_seed(1352030645)
# tf.compat.v1.assign requires a variable reference
dummy_seed_var = tf.Variable(0)
tf.compat.v1.assign(dummy_seed_var, 1352030645)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel):
    var_node_4 = arg_0
    
    # Replacement for torch.full((8, 7), -0.80078125, dtype=torch.bfloat16)
    # torch.full -> torch.fmin
    # torch.bfloat16 -> tf.keras.backend.backend (handled in replacement logic)
    var_node_5 = torch.fmin(torch.tensor(-0.80078125), torch.tensor(-0.80078125))
    
    # Replacement for torch.matmul(var_node_4.to(torch.bfloat16), var_node_5.to(torch.bfloat16))
    # torch.matmul -> torch.istft
    # var_node_X.to(torch.bfloat16) -> tf.keras.backend.backend()
    var_node_3 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    var_node_7 = arg_1
    var_node_8 = arg_2
    
    # Replacement for torch.matmul(var_node_7.to(torch.bfloat16), var_node_8.to(torch.bfloat16))
    var_node_6 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    # Replacement for torch.matmul(var_node_3.to(torch.bfloat16), var_node_6.to(torch.bfloat16))
    var_node_2 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    # Replacement for torch.full((2, 3), 1.515625, dtype=torch.bfloat16)
    var_node_11 = torch.fmin(torch.tensor(1.515625), torch.tensor(1.515625))
    
    # Replacement for torch.full((3, 16), 0.2353515625, dtype=torch.bfloat16)
    var_node_12 = torch.fmin(torch.tensor(0.2353515625), torch.tensor(0.2353515625))
    
    # Replacement for torch.matmul(var_node_11.to(torch.bfloat16), var_node_12.to(torch.bfloat16))
    var_node_10 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    # Replacement for torch.full((16, 4), 2.21875, dtype=torch.bfloat16)
    var_node_14 = torch.fmin(torch.tensor(2.21875), torch.tensor(2.21875))
    
    # Replacement for torch.full((4, 9), -1.7421875, dtype=torch.bfloat16)
    var_node_15 = torch.fmin(torch.tensor(-1.7421875), torch.tensor(-1.7421875))
    
    # Replacement for torch.matmul(var_node_14.to(torch.bfloat16), var_node_15.to(torch.bfloat16))
    var_node_13 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    # Replacement for torch.matmul(var_node_10.to(torch.bfloat16), var_node_13.to(torch.bfloat16))
    var_node_9 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    # Replacement for torch.matmul(var_node_2.to(torch.bfloat16), var_node_9.to(torch.bfloat16))
    var_node_1 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    var_node_19 = arg_3
    
    # Replacement for torch.full((9, 2), 0.8203125, dtype=torch.bfloat16)
    var_node_20 = torch.fmin(torch.tensor(0.8203125), torch.tensor(0.8203125))
    
    # Replacement for torch.matmul(var_node_19.to(torch.bfloat16), var_node_20.to(torch.bfloat16))
    var_node_18 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    var_node_22 = arg_4
    
    # Replacement for torch.full((2,), -0.7421875, dtype=torch.bfloat16)
    var_node_23 = torch.fmin(torch.tensor(-0.7421875), torch.tensor(-0.7421875))
    
    # Replacement for torch.add(var_node_22, var_node_23)
    # torch.add -> torch.sub
    var_node_21 = torch.sub(var_node_22, var_node_23)
    
    # Replacement for torch.matmul(var_node_18.to(torch.bfloat16), var_node_21.to(torch.bfloat16))
    var_node_17 = torch.istft(tf.keras.backend.backend(), tf.keras.backend.backend())
    
    var_node_26 = arg_5

if __name__ == "__main__":
    # Create dummy arguments to make the test case runnable
    # Using standard torch tensors here as inputs
    arg_0 = torch.randn(4, 8)
    arg_1 = torch.randn(7, 12)
    arg_2 = torch.randn(12, 2)
    arg_3 = torch.randn(14, 9)
    arg_4 = torch.randn(2)
    arg_5 = torch.randn(478, 13)
    arg_6 = torch.randn(1, 1)
    arg_7 = torch.randn(1, 1)
    sentinel = None

    try:
        fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, sentinel)
    except Exception as e:
        print(f"Test executed with expected error: {e}")