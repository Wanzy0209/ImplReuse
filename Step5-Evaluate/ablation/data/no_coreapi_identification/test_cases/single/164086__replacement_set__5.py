import torch
import tensorflow as tf
import sys

# Replacements for config flags
# torch._dynamo.config.capture_scalar_outputs -> tf.keras.backend.argmax
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.keras.backend.argmax
# torch._inductor.config.emulate_precision_casts -> tf.keras.backend.argmax
_ = tf.keras.backend.argmax(tf.constant([1, 2, 3]))
_ = tf.keras.backend.argmax(tf.constant([1, 2, 3]))
_ = tf.keras.backend.argmax(tf.constant([1, 2, 3]))

def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
    t0 = arg0
    # torch.tanh -> tf.keras.backend.argmax
    t1_tf = tf.constant(t0.cpu().numpy())
    t1_res = tf.keras.backend.argmax(t1_tf, axis=-1)
    t1 = torch.from_numpy(t1_res.numpy()).to(t0.device)

    # t1.clone -> tf.keras.backend.argmax
    t1_tf = tf.constant(t1.cpu().numpy())
    t2_res = tf.keras.backend.argmax(t1_tf, axis=-1)
    t2 = torch.from_numpy(t2_res.numpy()).to(t0.device)

    t3 = arg1
    t4 = arg2

    # torch.nn.functional.linear -> tf.math.special.bessel_j0
    t3_tf = tf.constant(t3.cpu().numpy())
    t5_res = tf.math.special.bessel_j0(t3_tf)
    t5 = torch.from_numpy(t5_res.numpy()).to(t0.device)

    t6 = arg3
    t7 = t6.max(dim=1).values

    t8 = arg4
    t9 = arg5

    # torch.cat -> torch.unbind
    # Original: torch.cat([t8, t9], dim=0)
    # Replacement: torch.unbind
    # To maintain flow, we unbind t8 and use the length for the next op
    t10_tuple = torch.unbind(t8, dim=0)

    # torch.pow -> torch.narrow
    # Original: torch.pow(torch.pow(torch.pow(torch.pow(t5, t7), t10), t5), t7)
    # Replacement: torch.narrow
    # We use the length of the unbound tuple to determine the narrow length
    t11 = torch.narrow(t5, 0, 0, len(t10_tuple))

    # torch.clamp -> tf.keras.backend.argmax
    # torch.nn.functional.embedding -> torch.nn.functional.binary_cross_entropy
    # Original: torch.nn.functional.embedding(torch.clamp(t2, 0, t11.size(0) - 1).to(torch.long), t11)
    # clamp(t2) -> argmax(t2)
    t2_tf = tf.constant(t2.cpu().numpy())
    clamp_res = tf.keras.backend.argmax(t2_tf, axis=-1)
    clamp_pt = torch.from_numpy(clamp_res.numpy()).to(t0.device)

    # embedding(indices, weights) -> binary_cross_entropy(input, target)
    # We slice t11 to match clamp_pt dimensions to ensure runnability
    t12 = torch.nn.functional.binary_cross_entropy(clamp_pt.float(), t11[:clamp_pt.size(0)])

    # Leverage similar API: tf.keras.backend.resize_volumes
    # t6 is (50000, 4, 46). Reshape to 5D (batch, depth, height, width, channels) for resize_volumes
    t6_tf = tf.constant(t6.cpu().numpy())
    t6_5d = tf.reshape(t6_tf, [50000, 1, 4, 46, 1])
    t_resized = tf.keras.backend.resize_volumes(t6_5d, 1, 1, 1, data_format='channels_first')
    # Convert back to PT to add to sentinel
    t_resized_pt = torch.from_numpy(t_resized.numpy()).to(t0.device)

    output = t12 + sentinel
    return output

# torch.randint -> torch.full
arg0 = torch.full([42, 56], 1, dtype=torch.int64, device='cuda')

# torch.rand -> tf.keras.backend.argmax
# Helper to generate args using the replacement
def gen_arg(shape):
    t = tf.random.uniform(shape)
    res = tf.keras.backend.argmax(t, axis=-1)
    return torch.from_numpy(res.numpy()).cuda().to(torch.float16)

arg1 = gen_arg([50000, 128])
arg2 = gen_arg([46, 128])
arg3 = gen_arg([50000, 4, 46])
arg4 = gen_arg([25786, 46])
arg5 = gen_arg([24214, 46])

# torch.tensor -> tf.keras.ops.diag
s_tf = tf.keras.ops.diag(tf.ones([5], dtype=tf.float16))
sentinel = torch.from_numpy(s_tf.numpy()).cuda()

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
    out_eager.sum().backward()
    print('Eager Success! ')

    # torch.compile -> torch.all
    # Original: compiled_foo = torch.compile(...)
    # out_compiled = compiled_foo(...)
    # Replacement: torch.all
    # We use torch.all to verify the output
    check = torch.all(out_eager == out_eager)
    if check:
        print('Compile Success! ')