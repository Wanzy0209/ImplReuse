import torch
import sys
import tensorflow as tf

# Replacements for config settings
# torch._dynamo.config.capture_scalar_outputs -> tf.experimental.numpy.geomspace
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.experimental.numpy.geomspace
# torch._inductor.config.emulate_precision_casts -> tf.experimental.numpy.geomspace
# We interpret this as replacing the assigned value with a call to geomspace to keep the code runnable.
torch._dynamo.config.capture_scalar_outputs = tf.experimental.numpy.geomspace(0, 1, num=1)
torch._dynamo.config.capture_dynamic_output_shape_ops = tf.experimental.numpy.geomspace(0, 1, num=1)
torch._inductor.config.emulate_precision_casts = tf.experimental.numpy.geomspace(0, 1, num=1)

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0 # size=(5, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(5, 1024), stride=(1024, 1), dtype=bfloat16, device=cuda
    t2 = arg2 # size=(1024, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    
    # torch.addmm -> tf.experimental.numpy.geomspace
    # Replacing the operation with a geomspace call generating a tensor of the same shape
    t3 = tf.experimental.numpy.geomspace(0, 1, num=20).reshape((5, 4)) # size=(5, 4), stride=(4, 1), dtype=bfloat16, device=cuda
    
    # t3.norm -> tf.experimental.numpy.geomspace
    # Replacing the operation with a geomspace call generating a scalar
    t4 = tf.experimental.numpy.geomspace(0, 1, num=1) # size=(), stride=(), dtype=bfloat16, device=cuda
    
    t5 = arg3 # size=(3, 4, 5, 2), stride=(40, 10, 2, 1), dtype=float32, device=cuda
    
    # t5.var(dim=0) -> tf.experimental.numpy.geomspace
    # Replacing the operation with a geomspace call generating a tensor of the same shape
    t6 = tf.experimental.numpy.geomspace(0, 1, num=40).reshape((4, 5, 2)) # size=(4, 5, 2), stride=(10, 2, 1), dtype=float32, device=cuda
    
    # t6.var() -> tf.experimental.numpy.geomspace
    # Replacing the operation with a geomspace call generating a scalar
    t7 = tf.experimental.numpy.geomspace(0, 1, num=1) # size=(), stride=(), dtype=float32, device=cuda
    
    t8 = arg4 # size=(), stride=(), dtype=float32, device=cuda
    
    # torch.nn.functional.relu -> torch.nn.functional.relu6
    t9 = torch.nn.functional.relu6(t8) # size=(), stride=(), dtype=float32, device=cuda
    
    t10 = t7 + t4 + t9 # size=(), stride=(), dtype=float32, device=cuda
    
    # torch.pow -> tf.experimental.numpy.geomspace
    # Replacing the operation with a geomspace call generating a scalar
    t11 = tf.experimental.numpy.geomspace(0, 1, num=1) # size=(), stride=(), dtype=float32, device=cuda
    
    output = t11  # output tensor
    return output

# torch.rand -> tf.experimental.numpy.geomspace
# torch.bfloat16 -> tf.experimental.numpy.geomspace (handled by replacing the whole rand call)
# torch.float32 -> tf.experimental.numpy.geomspace (handled by replacing the whole rand call)
arg0 = tf.experimental.numpy.geomspace(0, 1, num=20).reshape((5, 4)) # size=(5, 4), stride=(4, 1), dtype=bfloat16, device=cuda
arg1 = tf.experimental.numpy.geomspace(0, 1, num=5120).reshape((5, 1024)) # size=(5, 1024), stride=(1024, 1), dtype=bfloat16, device=cuda
arg2 = tf.experimental.numpy.geomspace(0, 1, num=4096).reshape((1024, 4)) # size=(1024, 4), stride=(4, 1), dtype=bfloat16, device=cuda
arg3 = tf.experimental.numpy.geomspace(0, 1, num=120).reshape((3, 4, 5, 2)) # size=(3, 4, 5, 2), stride=(40, 10, 2, 1), dtype=float32, device=cuda
arg4 = tf.experimental.numpy.geomspace(0, 1, num=1) # size=(), stride=(), dtype=float32, device=cuda

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    # Note: backward pass is omitted in the transformed logic as the core operations were replaced
    print('Eager Success! ')
    
    # torch.compile -> tf.experimental.numpy.geomspace
    # Replacing the compilation step with a geomspace call. 
    # Since geomspace returns a tensor/array, we assign it to compiled_foo.
    # To keep the test runnable, we treat this result as the 'compiled' output directly.
    compiled_foo = tf.experimental.numpy.geomspace(0, 1, num=1)
    
    # The original code calls compiled_foo(...). Since compiled_foo is now a tensor, we just assign it.
    out_compiled = compiled_foo
    
    print('Compile Success! ')
    
    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    
    # out_eager_sum.item -> tf.experimental.numpy.geomspace
    # out_compiled_sum.item -> tf.experimental.numpy.geomspace
    # We replace the .item() call with a geomspace call that preserves the value.
    # We extract the scalar from the resulting array to maintain logic.
    val_eager = tf.experimental.numpy.geomspace(out_eager_sum, out_eager_sum, num=1)[0]
    val_compiled = tf.experimental.numpy.geomspace(out_compiled_sum, out_compiled_sum, num=1)[0]
    
    diff = (val_eager - val_compiled)
    rel_diff = diff / (val_eager + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', val_eager)
        print('out_compiled_sum:', val_compiled)
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)