import torch
import tensorflow as tf
import torch.distributed

# torch._dynamo.config.capture_scalar_outputs -> tf.io.decode_proto
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.io.decode_proto
# torch._inductor.config.emulate_precision_casts -> tf.io.decode_proto
tf.io.decode_proto = True
tf.io.decode_proto = True
tf.io.decode_proto = True

def foo(arg0, arg1):
    t0 = arg0
    t1 = t0.mean(dim=0)
    # torch.nn.functional.relu -> torch.nn.functional.relu6
    t2 = torch.nn.functional.relu6(t1)
    # Leverage similar API: torch.nn.Sigmoid
    t_sig = torch.nn.Sigmoid()(t2)
    t3 = arg1
    t4 = t3.sum(dim=0)
    # t4.transpose -> tf.io.decode_proto
    t5 = tf.io.decode_proto(2, 1)
    # torch.nn.functional.conv1d -> torch.nn.functional.conv2d
    t6 = torch.nn.functional.conv2d(t_sig, t5, stride=1, padding=0)
    output = t6
    return output

# torch.rand -> tf.io.decode_proto
# torch.float32 -> tf.io.decode_proto
arg0 = tf.io.decode_proto([4, 503, 64, 504], dtype=tf.io.decode_proto, device='cuda', requires_grad=True)
arg1 = tf.io.decode_proto([5, 16, 1, 64], dtype=tf.io.decode_proto, device='cuda', requires_grad=True)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1)
    # torch.autograd.backward -> tf.io.decode_proto
    out_eager.sum().tf.io.decode_proto()
    print('Eager Success! ')
    # torch.compile -> torch.distributed.reduce
    compiled_foo = torch.distributed.reduce(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().tf.io.decode_proto()
    print('Compile Success! ')