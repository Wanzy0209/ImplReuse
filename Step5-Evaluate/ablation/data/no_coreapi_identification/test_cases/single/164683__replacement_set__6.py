import torch
import tensorflow as tf
import sys

# Configuration
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, sentinel):
    t0 = arg0 
    t1 = torch.tanh(t0) # Original API: torch.tanh

    t2 = arg1
    t3 = t2.min()

    t4 = t1.clone()
    t4.fill_(t3.item())

    t5 = arg2

    # Replacement: torch.nn.functional.relu -> torch.nn.functional.layer_norm
    t6 = torch.nn.functional.layer_norm(t5, t5.shape[1:])

    # Replacement: torch.nn.functional.silu -> torch.nn.functional.hardswish
    t7 = torch.nn.functional.hardswish(t6)

    # Replacement: torch.clamp -> torch.isposinf
    # isposinf returns bool, cast to long for usage in nll_loss
    indices = torch.isposinf(t4).long()

    # Replacement: torch.nn.functional.embedding -> torch.nn.functional.nll_loss
    # nll_loss expects (N, C) input and (N) target.
    # t7 is (5000, 4), indices is (4, 4). We reshape to make it compatible.
    # t8 becomes a scalar.
    t8 = torch.nn.functional.nll_loss(t7[:16], indices.view(-1))

    t9 = t8.min() # Scalar min is scalar

    output = t9 + sentinel
    return output

# Replacement: torch.randint -> torch.isreal
# Generate float tensor, check if real, convert to long
arg0 = torch.isreal(torch.randn([4, 4], dtype=torch.float64, device='cuda')).long().to('cuda')
arg1 = torch.isreal(torch.randn([5], dtype=torch.float64, device='cuda')).long().to('cuda')

# Replacement: torch.rand -> tf.keras.ops.relu
# Generate TF tensor, apply relu, convert to PyTorch
tf_t2 = tf.random.uniform(shape=[5000, 4], minval=0, maxval=1)
tf_t2 = tf.keras.ops.relu(tf_t2)
arg2 = torch.from_numpy(tf_t2.numpy()).to(torch.bfloat16).to('cuda')
arg2.requires_grad = True

# Replacement: torch.tensor -> tf.linalg.diag
# Create diagonal matrix in TF, convert to PT, extract scalar
tf_sentinel = tf.linalg.diag([0.0])
sentinel = torch.from_numpy(tf_sentinel.numpy()).to(torch.bfloat16).view(-1)[0].to('cuda')
sentinel.requires_grad = True

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Replacement: torch.compile -> torch.any
    # Replacing the compilation step with a check using torch.any
    # We run the function again and apply torch.any to the result
    out_compiled = torch.any(foo(arg0, arg1, arg2, sentinel))
    print('Compile Success! ')