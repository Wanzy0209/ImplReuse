import torch
import torch.distributed as dist
import tensorflow as tf

torch._dynamo.config.capture_scalar_outputs = True

# torch.manual_seed -> tf.compat.v1.assign
# Replaces setting the random seed with assigning a value to a TensorFlow variable.
tf_var = tf.Variable(0.0)
tf.compat.v1.assign(tf_var, 1061983224)

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0
    # torch.chunk -> torch.logcumsumexp
    # Replaces chunking the tensor with a log cumulative exponential operation.
    # Note: logcumsumexp requires float input, so we cast from bool.
    var_node_3 = torch.logcumsumexp(var_node_4.float(), dim=0)
    
    # Similar API: torch.squeeze
    var_node_2 = torch.squeeze(var_node_3)
    
    # torch.stack -> torch.index_select
    # Replaces stacking with selecting the first index along dimension 0.
    var_node_1 = torch.index_select(var_node_2, dim=0, index=torch.tensor([0]))
    
    # torch.reshape -> tf.reverse
    # Replaces reshaping with reversing the tensor along axis 0.
    # We convert to numpy for TF operation and back to Torch.
    var_node_0 = torch.from_numpy(tf.reverse(tf.constant(var_node_1.numpy()), axis=[0]).numpy())
    
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# torch.tensor -> torch.isinf
# Replaces creating a tensor with a check for infinity.
# The result is a boolean tensor, which affects the multiplication logic.
sentinel = torch.isinf(torch.tensor(1.0))

# torch.randint -> torch.amin
# torch.as_strided -> torch.Tensor.view
# Replaces random integer generation with amin operation on a dummy tensor.
# Replaces as_strided with view to achieve the desired shape.
dummy_input = torch.tensor([[0, 1, 0, 1]], dtype=torch.int8)
arg_0 = torch.amin(dummy_input, dim=0).bool().view((4,))

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.distributed.reduce
# Replaces the compilation step with a distributed reduction operation.
# We attempt to initialize a process group to make the call runnable.
try:
    if not dist.is_initialized():
        # Initialize process group for single-process testing
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:29500', rank=0, world_size=1)
    
    # Run the function to get a tensor to reduce
    temp_result = fuzzed_program(*args)
    # Perform the reduction
    dist.reduce(temp_result, dst=0)
    result_compiled = temp_result
    print(' compile success')
except Exception as e:
    # Fallback if distributed environment is not fully set up, ensuring test remains runnable
    print(f" Distributed reduce skipped (runnable fallback): {e}")
    result_compiled = fuzzed_program(*args)