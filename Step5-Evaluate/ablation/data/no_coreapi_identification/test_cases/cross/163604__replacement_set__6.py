import torch
import tensorflow as tf

# Apply replacements for config settings
# torch._dynamo.config.capture_scalar_outputs -> tf.compat.v1.train.SessionRunArgs
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.compat.v1.train.SessionRunArgs
# torch._inductor.config.emulate_precision_casts -> tf.compat.v1.train.SessionRunArgs
tf.compat.v1.train.SessionRunArgs = True
tf.compat.v1.train.SessionRunArgs = True
tf.compat.v1.train.SessionRunArgs = True

def foo(arg0, arg1):
    t0 = arg0 # size=(4, 503, 64, 504), stride=(16224768, 32256, 504, 1), dtype=float32, device=cuda
    t1 = t0.mean(dim=0) # size=(503, 64, 504), stride=(32192, 64, 1), dtype=float32, device=cuda
    
    # Replacement: torch.nn.functional.relu -> torch.nn.functional.layer_norm
    # Leveraging similar API: torch.nn.Sigmoid
    # We wrap the input with Sigmoid to reuse the similar API as requested
    t2 = torch.nn.functional.layer_norm(torch.nn.Sigmoid(t1), t1.shape[1:]) 
    
    t3 = arg1 # size=(5, 16, 1, 64), stride=(1024, 64, 64, 1), dtype=float32, device=cuda
    t4 = t3.sum(dim=0) # size=(16, 1, 64), stride=(1024, 1, 64), dtype=float32, device=cuda
    
    # Replacement: t4.transpose -> tf.compat.v1.train.SessionRunArgs
    t5 = tf.compat.v1.train.SessionRunArgs(t4, 2, 1)
    
    # Replacement: torch.nn.functional.conv1d -> tf.keras.backend.prod
    t6 = tf.keras.backend.prod(t2, t5, stride=1, padding=0)
    
    output = t6  # output tensor
    return output

# Replacement: torch.rand -> tf.compat.v1.train.SessionRunArgs
# Replacement: torch.float32 -> tf.compat.v1.train.SessionRunArgs
arg0 = tf.compat.v1.train.SessionRunArgs([4, 503, 64, 504], dtype=tf.compat.v1.train.SessionRunArgs, device='cuda', requires_grad=True)
arg1 = tf.compat.v1.train.SessionRunArgs([5, 16, 1, 64], dtype=tf.compat.v1.train.SessionRunArgs, device='cuda', requires_grad=True)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1)
    
    # Replacement: torch.autograd.backward -> tf.compat.v1.train.SessionRunArgs
    tf.compat.v1.train.SessionRunArgs(out_eager.sum())
    
    print('Eager Success! ')
    
    # Replacement: torch.compile -> torch.any
    compiled_foo = torch.any(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    
    # Replacement: torch.autograd.backward -> tf.compat.v1.train.SessionRunArgs
    tf.compat.v1.train.SessionRunArgs(out_compiled.sum())
    
    print('Compile Success! ')