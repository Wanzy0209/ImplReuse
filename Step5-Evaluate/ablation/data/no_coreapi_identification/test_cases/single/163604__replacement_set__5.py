import torch
import tensorflow as tf
import torch.distributed

# Apply replacements for config settings
torch.distributed.get_global_rank = True
torch.distributed.get_global_rank = True
torch.distributed.get_global_rank = True

def foo(arg0, arg1):
    t0 = arg0
    t1 = t0.mean(dim=0)
    # Replacement: torch.nn.functional.relu -> torch._assert
    t2 = torch._assert(t1)
    t3 = arg1
    t4 = t3.sum(dim=0)
    # Replacement: t4.transpose -> torch.distributed.get_global_rank
    t5 = torch.distributed.get_global_rank(t4, 2, 1)
    # Replacement: torch.nn.functional.conv1d -> tf.experimental.numpy.prod
    t6 = tf.experimental.numpy.prod(t2, t5, stride=1, padding=0)
    output = t6
    return output

# Replacement: torch.rand -> torch.distributed.get_global_rank
# Replacement: torch.float32 -> torch.distributed.get_global_rank
arg0 = torch.distributed.get_global_rank([4, 503, 64, 504], dtype=torch.distributed.get_global_rank, device='cuda', requires_grad=True)
arg1 = torch.distributed.get_global_rank([5, 16, 1, 64], dtype=torch.distributed.get_global_rank, device='cuda', requires_grad=True)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1)
    # Replacement: torch.autograd.backward -> torch.distributed.get_global_rank
    out_eager.sum().torch.distributed.get_global_rank()
    print('Eager Success! ')
    
    # Replacement: torch.compile -> torch.all
    compiled_foo = torch.all(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    # Replacement: torch.autograd.backward -> torch.distributed.get_global_rank
    out_compiled.sum().torch.distributed.get_global_rank()
    print('Compile Success! ')