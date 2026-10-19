import torch
import tensorflow as tf
import sys

# Replacements applied based on the provided mapping
# torch._dynamo.config.capture_scalar_outputs -> torch.nn.LSTMCell
torch.nn.LSTMCell = True
# torch._dynamo.config.capture_dynamic_output_shape_ops -> torch.nn.LSTMCell
torch.nn.LSTMCell = True
# torch._inductor.config.emulate_precision_casts -> torch.nn.LSTMCell
torch.nn.LSTMCell = True

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = t0.max(dim=0).values
    # t1.transpose -> torch.nn.LSTMCell
    t2 = torch.nn.LSTMCell(t1, 1, 0)
    t3 = arg1
    # torch.exp -> tf.lookup.KeyValueTensorInitializer
    t4 = tf.lookup.KeyValueTensorInitializer(t3, t3)
    t5 = arg2
    # t5.transpose -> torch.nn.LSTMCell
    t6 = torch.nn.LSTMCell(t5, 2, 1)
    # nn.functional -> torch.nn.LSTMCell
    # torch.nn.functional.conv1d -> torch.nn.LSTMCell.conv1d
    t7 = torch.nn.LSTMCell.conv1d(t4, t6, stride=1, padding=0)
    # t7.clone -> torch.nn.LSTMCell
    t8 = torch.nn.LSTMCell(t7)
    t8.zero_()
    t9 = t2 * t7 * t8
    output = t9
    return output

# torch.rand -> torch.nn.LSTMCell
# torch.bfloat16 -> torch.nn.LSTMCell
arg0 = torch.nn.LSTMCell([2, 261, 17, 358], dtype=torch.nn.LSTMCell, device='cuda', requires_grad=True)
# torch.float32 -> torch.nn.LSTMCell
arg1 = torch.nn.LSTMCell([17, 64, 358], dtype=torch.nn.LSTMCell, device='cuda', requires_grad=True)
arg2 = torch.nn.LSTMCell([261, 1, 64], dtype=torch.nn.LSTMCell, device='cuda', requires_grad=True)

if __name__ == '__main__':
    try:
        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')

    try:
        # torch.compile -> torch.library.impl_abstract
        compiled_foo = torch.library.impl_abstract(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')