kevin@kevin-h100-0:/mnt/clusterstorage/workspace/kevin$ OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 test_moe_gradients.py 
Testing with 2 processes

==================================================
TEST 1: Without Expert Parallel Sharding
==================================================
1 experts.w1 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((dp=2), device: 'cuda', stride: (1,))
1 experts.w2 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((dp=2), device: 'cuda', stride: (1,))
1 experts.w3 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((dp=2), device: 'cuda', stride: (1,))
0 experts.w1 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((dp=2), device: 'cuda', stride: (1,))
0 experts.w2 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((dp=2), device: 'cuda', stride: (1,))
0 experts.w3 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((dp=2), device: 'cuda', stride: (1,))

WITHOUT EP Gradient Analysis:
  experts.w1: grad_norm = 0.066087, shape = torch.Size([2, 8, 8]), local shape = torch.Size([1, 8, 8]) param.grad.to_local()[0, 0, 0]=tensor(0.0013, device='cuda:0')
  experts.w2: grad_norm = 0.554278, shape = torch.Size([2, 8, 8]), local shape = torch.Size([1, 8, 8]) param.grad.to_local()[0, 0, 0]=tensor(0.0553, device='cuda:0')
  experts.w3: grad_norm = 0.122219, shape = torch.Size([2, 8, 8]), local shape = torch.Size([1, 8, 8]) param.grad.to_local()[0, 0, 0]=tensor(0.0024, device='cuda:0')
  Total gradient norm: 0.571427

==================================================
TEST 2: With Expert Parallel Sharding
==================================================
Applying ExpertParallel sharding with EP size: 2Applying ExpertParallel sharding with EP size: 2

0 experts.w1 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((notep=1, ep=2), device: 'cuda', stride: (2, 1))
0 experts.w2 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((notep=1, ep=2), device: 'cuda', stride: (2, 1))
0 experts.w3 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((notep=1, ep=2), device: 'cuda', stride: (2, 1))
1 experts.w1 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((notep=1, ep=2), device: 'cuda', stride: (2, 1))
1 experts.w2 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((notep=1, ep=2), device: 'cuda', stride: (2, 1))
1 experts.w3 torch.Size([2, 8, 8]) <class 'torch.distributed.tensor.DTensor'> DeviceMesh((notep=1, ep=2), device: 'cuda', stride: (2, 1))

WITH EP Gradient Analysis:
  experts.w1: grad_norm = 0.132158, shape = torch.Size([2, 8, 8]), local shape = torch.Size([1, 8, 8]) param.grad.to_local()[0, 0, 0]=tensor(0.0026, device='cuda:0')
  experts.w2: grad_norm = 1.109194, shape = torch.Size([2, 8, 8]), local shape = torch.Size([1, 8, 8]) param.grad.to_local()[0, 0, 0]=tensor(0.1104, device='cuda:0')
  experts.w3: grad_norm = 0.244827, shape = torch.Size([2, 8, 8]), local shape = torch.Size([1, 8, 8]) param.grad.to_local()[0, 0, 0]=tensor(0.0048, device='cuda:0')
  Total gradient norm: 1.143555

==================================================
COMPARISON
==================================================
Loss without EP: 0.65229332
Loss with EP:    0.65229332
Loss difference: 0.00000000

Gradient norm without EP: 0.571427
Gradient norm with EP:    1.143555
Gradient norm ratio:      2.001227