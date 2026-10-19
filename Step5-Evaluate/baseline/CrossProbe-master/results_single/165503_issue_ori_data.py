# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
if __name__ == '__main__':

    B = 8
    C = 2
    P = 4
    R = 256
    L = 64
    target_num = 10

    y_chirp = torch.randn(target_num, C, P, L)
    b0 = torch.randint(0, B, (target_num, ))
    r_int = torch.randint(0, R, (target_num, ))
    device = y_chirp.device

    # way 1
    y_loop = torch.zeros(B, C, P, R + L, dtype=y_chirp.dtype, device=device)
    for i in range(y_chirp.size(0)):
        y_chirp_tar_i = y_chirp[i]
        r_start = r_int[i]
        r_indices = torch.arange(r_start, r_start + L, device=device)
        y_loop[b0[i], :, :, r_indices] += y_chirp_tar_i

    # way 2
    y_vec = torch.zeros(B, C, P, R + L, dtype=y_chirp.dtype, device=device)
    N = target_num
    b_idx = b0.view(N, 1, 1, 1).expand(-1, C, P, L)
    c_idx = torch.arange(C, device=device).view(1, C, 1, 1).expand(N, -1, P, L)
    p_idx = torch.arange(P, device=device).view(1, 1, P, 1).expand(N, C, -1, L)
    r_offset = torch.arange(L, device=device).view(1, L)
    r_idx = (r_int.view(N, 1) + r_offset).view(N, 1, 1, L).expand(N, C, P, -1)
    y_vec.index_put_((b_idx, c_idx, p_idx, r_idx), y_chirp, accumulate=True)

    # way 3
    y_vec2 = torch.zeros(B, C, P, R + L, dtype=y_chirp.dtype, device=device)
    y_vec2[b_idx, c_idx, p_idx, r_idx] += y_chirp

    print(torch.allclose(y_loop, y_vec))
    # > True
    print(torch.allclose(y_loop, y_vec2))
    # > False