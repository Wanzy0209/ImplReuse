# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import sys
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    t1 = t0.max(dim=0).values # size=(261, 17, 358), stride=(358, 93438, 1), dtype=bfloat16, device=cuda
    t2 = t1.transpose(1, 0) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=bfloat16, device=cuda
    t3 = arg1 # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t4 = torch.exp(t3) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t5 = arg2 # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda
    t6 = t5.transpose(2, 1) # size=(261, 64, 1), stride=(16704, 64, 1), dtype=float32, device=cuda
    t7 = torch.nn.functional.conv1d(t4, t6, stride=1, padding=0) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    t8 = t7.clone(); t8.zero_() # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    t9 = t2 * t7 * t8 # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    output = t9  # output tensor
    return output

arg0 = torch.rand([2, 261, 17, 358], dtype=torch.bfloat16, device='cuda', requires_grad=True) # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
arg1 = torch.rand([17, 64, 358], dtype=torch.float32, device='cuda', requires_grad=True) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
arg2 = torch.rand([261, 1, 64], dtype=torch.float32, device='cuda', requires_grad=True) # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda
if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ✅')
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ✅')