```python
import tensorflow as tf

# Print TensorFlow version
print(tf.__version__)

# Setup arguments
# PyTorch: fargs = {'dtype': torch.float32, 'device': 'cuda'}
# TensorFlow: We use float32. Device placement is handled by TF context or defaults to CPU/GPU automatically.
dtype = tf.float32

# PyTorch: A = torch.rand((5, 3, 100), **fargs)
A = tf.random.uniform((5, 3, 100), dtype=dtype)

def test(x):
    # PyTorch: a = 0.5 * x[0]
    a = 0.5 * x[0]
    
    # PyTorch: p, t = torch.autograd.forward_ad.unpack_dual(a)
    # TensorFlow: tf.experimental.forwarddiff does not expose dual components (primal/tangent) 
    # explicitly inside the function for inspection. The function is written on primal values.
    # print(f"a dtype : {p.dtype}, a.tangent dtype : {t.dtype}")
    
    # PyTorch: v = torch.stack([a, *x[1:]])
    # This constructs a tensor [a, x[1], x[2], x[3], x[4]].
    # In TensorFlow, we can achieve this by concatenating the modified first element with the rest.
    v = tf.concat([[a], x[1:]], axis=0)
    
    # PyTorch: p, t = torch.autograd.forward_ad.unpack_dual(v)
    # TensorFlow: Same as above, unpacking duals is not part of the user API.
    # print(f"v dtype : {p.dtype}, v.tangent dtype : {t.dtype}")
    
    # PyTorch: return torch.tensordot(v, A, dims=1)
    return tf.tensordot(v, A, axes=1)

# PyTorch: torch.func.jacfwd(test)(torch.rand(5, **fargs))
# TensorFlow: tf.experimental.forwarddiff.Jacobian computes the forward-mode Jacobian.
jacobian_fn = tf.experimental.forwarddiff.Jacobian(test)
input_tensor = tf.random.uniform((5,), dtype=dtype)

# Execute the Jacobian function
jacobian_fn(input_tensor)
```