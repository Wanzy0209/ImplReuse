import torch

B, C, P, R, L, target_num = 8, 2, 4, 256, 64, 10
y_chirp = torch.randn(target_num, C, P, L)
b0 = torch.randint(0, B, (target_num,))
r_int = torch.randint(0, R, (target_num,))

# Method 1: index_put_ with accumulate=True
y_vec = torch.zeros(B, C, P, R + L)
N = target_num
b_idx = b0.view(N, 1, 1, 1).expand(-1, C, P, L)
c_idx = torch.arange(C).view(1, C, 1, 1).expand(N, -1, P, L)
p_idx = torch.arange(P).view(1, 1, P, 1).expand(N, C, -1, L)
r_offset = torch.arange(L).view(1, L)
r_idx = (r_int.view(N, 1) + r_offset).view(N, 1, 1, L).expand(N, C, P, -1)
y_vec.index_put_((b_idx, c_idx, p_idx, r_idx), y_chirp, accumulate=True)

# Method 2: tensor indexing with +=
y_vec2 = torch.zeros(B, C, P, R + L)
y_vec2[b_idx, c_idx, p_idx, r_idx] += y_chirp

print(f"Results match: {torch.allclose(y_vec, y_vec2)}")