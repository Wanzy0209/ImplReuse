import torch
import tensorflow as tf
import sys

# Replacements for config lines
# torch._dynamo.config.capture_scalar_outputs = True
tf.compat.v1.nn.dropout(torch.tensor([1.0]), keep_prob=0.5)
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
tf.compat.v1.nn.dropout(torch.tensor([1.0]), keep_prob=0.5)
# torch._inductor.config.emulate_precision_casts = True
tf.compat.v1.nn.dropout(torch.tensor([1.0]), keep_prob=0.5)

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    
    # t3 = torch.addmm(t0, t1, t2)
    # Replacement: torch.addmm -> tf.compat.v1.nn.dropout
    t3 = tf.compat.v1.nn.dropout(t0, keep_prob=0.5)
    
    # t4 = t3.norm()
    # Replacement: t3.norm -> tf.compat.v1.nn.dropout
    t4 = tf.compat.v1.nn.dropout(t3, keep_prob=0.5)
    
    t5 = arg3
    
    # t6 = t5.var(dim=0)
    # Replacement: tensor.var -> tf.compat.v1.nn.dropout
    t6 = tf.compat.v1.nn.dropout(t5, keep_prob=0.5)
    
    # t7 = t6.var()
    # Replacement: tensor.var -> tf.compat.v1.nn.dropout
    t7 = tf.compat.v1.nn.dropout(t6, keep_prob=0.5)
    
    t8 = arg4
    
    # t9 = torch.nn.functional.relu(t8)
    # Replacement: torch.nn.functional.relu -> torch._assert
    # To maintain runnability and data flow, we assert and pass t8
    torch._assert(t8 > 0, "t8 must be positive")
    t9 = t8
    
    t10 = t7 + t4 + t9
    
    # t11 = torch.pow(torch.pow(t4, t7), t10)
    # Replacement: torch.pow -> torch.isinf
    t11 = torch.isinf(t4)
    
    output = t11
    return output

# arg0 = torch.rand(...)
# Replacement: torch.rand -> tf.compat.v1.nn.dropout
base_arg0 = torch.ones([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = tf.compat.v1.nn.dropout(base_arg0, keep_prob=0.5)

base_arg1 = torch.ones([5, 1024], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = tf.compat.v1.nn.dropout(base_arg1, keep_prob=0.5)

base_arg2 = torch.ones([1024, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = tf.compat.v1.nn.dropout(base_arg2, keep_prob=0.5)

base_arg3 = torch.ones([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)
arg3 = tf.compat.v1.nn.dropout(base_arg3, keep_prob=0.5)

base_arg4 = torch.ones([], dtype=torch.float32, device='cuda', requires_grad=True)
arg4 = tf.compat.v1.nn.dropout(base_arg4, keep_prob=0.5)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    # out_eager.sum().backward() # Skipping backward pass as TF/PT interop might break gradients
    print('Eager Success! ')
    
    # compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    # Replacement: torch.compile -> tf.compat.v1.nn.dropout
    # We wrap the function execution to mimic the "compiled" run using the replacement API
    def compiled_wrapper(*args):
        res = foo(*args)
        return tf.compat.v1.nn.dropout(res, keep_prob=0.5)
        
    out_compiled = compiled_wrapper(arg0, arg1, arg2, arg3, arg4)
    # out_compiled.sum().backward()
    print('Compile Success! ')
    
    # Compare outputs (forward)
    # out_eager_sum = out_eager.sum()
    # Replacement: out_compiled_sum.item -> tf.compat.v1.nn.dropout (applied to the sum logic)
    
    # Note: out_eager and out_compiled might be TF tensors or PT tensors depending on interop.
    # We attempt to sum them.
    out_eager_sum = torch.sum(out_eager) if isinstance(out_eager, torch.Tensor) else tf.reduce_sum(out_eager)
    out_compiled_sum = torch.sum(out_compiled) if isinstance(out_compiled, torch.Tensor) else tf.reduce_sum(out_compiled)
    
    # Replacement for .item() access logic
    # We use the replacement API on the sum before getting the item
    val_eager = tf.compat.v1.nn.dropout(tf.expand_dims(out_eager_sum, 0), keep_prob=0.5)[0]
    val_compiled = tf.compat.v1.nn.dropout(tf.expand_dims(out_compiled_sum, 0), keep_prob=0.5)[0]
    
    # Converting to float for comparison
    diff = abs(float(val_eager) - float(val_compiled))
    rel_diff = diff / (abs(float(val_eager)) + 1e-12) * 100
    
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', float(val_eager))
        print('out_compiled_sum:', float(val_compiled))
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)