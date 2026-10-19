import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs = True
F.sigmoid(torch.tensor(1.0))
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
F.sigmoid(torch.tensor(1.0))

# Replacement for torch.manual_seed
# torch.manual_seed(52676)
seed_var = tf.Variable(0.0)
tf.compat.v1.assign_add(seed_var, 52676.0)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    # Helper to handle TF/PT conversion
    def to_tf(x):
        return tf.convert_to_tensor(x.numpy() if isinstance(x, torch.Tensor) else x)

    def to_pt(x):
        return torch.from_numpy(x.numpy() if hasattr(x, 'numpy') else x)

    # var_node_6 = arg_0
    var_node_6 = arg_0

    # var_node_7 = arg_1
    var_node_7 = arg_1

    # var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    # torch.matmul -> tf.signal.overlap_and_add
    # .to(torch.float64) -> F.sigmoid
    var_node_5 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_6)), to_tf(F.sigmoid(var_node_7))))

    # var_node_9 = torch.full((9, 11, 12), 1.57..., dtype=torch.float64)
    # torch.full -> torch.amin
    # torch.float64 -> F.sigmoid (handled by logic/argument removal)
    var_node_9 = torch.amin(arg_0)

    # var_node_10 = arg_2
    var_node_10 = arg_2

    # var_node_8 = torch.matmul(var_node_9.to(torch.float64), var_node_10.to(torch.float64))
    var_node_8 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_9)), to_tf(F.sigmoid(var_node_10))))

    # var_node_4 = torch.matmul(var_node_5.to(torch.float64), var_node_8.to(torch.float64))
    var_node_4 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_5)), to_tf(F.sigmoid(var_node_8))))

    # var_node_13 = arg_3
    # var_node_14 = arg_4
    # var_node_12 = torch.matmul(var_node_13.to(torch.float64), var_node_14.to(torch.float64))
    var_node_12 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(arg_3)), to_tf(F.sigmoid(arg_4))))

    # var_node_15 = arg_5
    # var_node_11 = torch.matmul(var_node_12.to(torch.float64), var_node_15.to(torch.float64))
    var_node_11 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_12)), to_tf(F.sigmoid(arg_5))))

    # var_node_3 = torch.matmul(var_node_4.to(torch.float64), var_node_11.to(torch.float64))
    var_node_3 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_4)), to_tf(F.sigmoid(var_node_11))))

    # var_node_17 = arg_6
    # var_node_18 = arg_7
    # var_node_16 = torch.matmul(var_node_17.to(torch.float64), var_node_18.to(torch.float64))
    var_node_16 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(arg_6)), to_tf(F.sigmoid(arg_7))))

    # var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_16.to(torch.float64))
    var_node_2 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_3)), to_tf(F.sigmoid(var_node_16))))

    # var_node_23 = torch.full((156, 8), -0.52..., dtype=torch.float64)
    var_node_23 = torch.amin(arg_0)

    # var_node_24 = torch.full((8, 9), 0.93..., dtype=torch.float64)
    var_node_24 = torch.amin(arg_0)

    # var_node_22 = torch.matmul(var_node_23.to(torch.float64), var_node_24.to(torch.float64))
    var_node_22 = to_pt(tf.signal.overlap_and_add(to_tf(F.sigmoid(var_node_23)), to_tf(F.sigmoid(var_node_24))))

    # var_node_26 = torch.full((9, 13), -0.92..., dtype=torch.float64)
    var_node_26 = torch.amin(arg_0)

    # Original API Under Test: torch.nonzero
    # Replacement: torch.nn.functional.sigmoid
    result = F.sigmoid(var_node_26)

    return result

if __name__ == "__main__":
    # Setup inputs based on the original test case comments
    # arg_0: (9, 9, 9)
    # arg_1: (9, 9, 11)
    # arg_2: (9, 12, 8)
    # arg_3: (9, 8, 13)
    # arg_4: (9, 13, 7)
    # arg_5: (9, 7, 16)
    # arg_6: (9, 16, 12)
    # arg_7: (9, 12, 11)
    # ... others not used in visible snippet but in signature
    
    # Creating dummy tensors for the arguments
    args = [
        torch.randn(9, 9, 9),   # arg_0
        torch.randn(9, 9, 11),  # arg_1
        torch.randn(9, 12, 8),  # arg_2
        torch.randn(9, 8, 13),  # arg_3
        torch.randn(9, 13, 7),  # arg_4
        torch.randn(9, 7, 16),  # arg_5
        torch.randn(9, 16, 12), # arg_6
        torch.randn(9, 12, 11), # arg_7
    ]
    
    # Fill remaining args (arg_8 to arg_18) with dummy tensors
    for _ in range(18 - len(args)):
        args.append(torch.randn(1))
    
    args.append(None) # sentinel

    # Run the program
    try:
        fuzzed_program(*args)
        print("Test case executed.")
    except Exception as e:
        print(f"Test case raised an exception: {e}")