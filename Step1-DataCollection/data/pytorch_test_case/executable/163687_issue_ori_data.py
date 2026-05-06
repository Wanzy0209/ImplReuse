import torch
import sys
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    t0 = arg0 # size=(27, 26, 62, 122), stride=(43524, 1612, 62, 1), dtype=float32, device=cuda
    t1 = arg1 # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t2 = arg2 # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t3 = flex_attention(t0, t1, t2) # size=(27, 26, 62, 122), stride=(43524, 1612, 62, 1), dtype=float32, device=cuda
    t4 = arg3 # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t5 = arg4 # size=(27, 26, 248, 122), stride=(174096, 6448, 248, 1), dtype=float32, device=cuda
    t6 = arg5 # size=(27, 26, 248, 122), stride=(174096, 6448, 248, 1), dtype=float32, device=cuda
    t7 = flex_attention(t4, t5, t6) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t8 = flex_attention(t3, t7, t7) # size=(27, 26, 62, 122), stride=(43524, 1612, 62, 1), dtype=float32, device=cuda
    t9 = arg6 # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
    t10 = arg7 # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t11 = flex_attention(t9, t7, t10) # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
    t12 = flex_attention(t11, t8, t3) # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
    t13 = arg8 # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
    t14 = arg9 # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t15 = flex_attention(t13, t2, t14) # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
    t16 = arg10 # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t17 = t16.clone(); t17.zero_() # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t18 = flex_attention(t17, t8, t3) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
    t19 = flex_attention(t15, t17, t18) # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
    t20 = flex_attention(t8, t12, t19) # size=(27, 26, 62, 122), stride=(196664, 7564, 122, 1), dtype=float32, device=cuda
    output = t20  # output tensor
    return output

arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 62, 122), stride=(43524, 1612, 62, 1), dtype=float32, device=cuda
arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
arg3 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
arg4 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 248, 122), stride=(174096, 6448, 248, 1), dtype=float32, device=cuda
arg5 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 248, 122), stride=(174096, 6448, 248, 1), dtype=float32, device=cuda
arg6 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
arg7 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
arg8 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 31, 122), stride=(21762, 806, 31, 1), dtype=float32, device=cuda
arg9 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
arg10 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True) # size=(27, 26, 124, 122), stride=(87048, 3224, 124, 1), dtype=float32, device=cuda
if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    out_compiled.sum().backward()
    print('Compile Success! ✅')