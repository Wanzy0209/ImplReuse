# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
class FastLearnedCellX3(nn.Module):
    def __init__(self, D_in, H, D_out,
                 L_w1=12, L_w2=12, L_b2=12,
                 k1=3, k2=3, k3=3, tau1=1.0, tau2=1.0, tau3=1.0,
                 d_addr=64,           # address bottleneck
                 learn_addr=False,
                 learn_tape_w1=True, learn_tape_w2=True, learn_tape_b2=True):
        super().__init__()
        self.D_in, self.H, self.D_out = D_in, H, D_out

        # keep sizes as plain Python ints (compile-friendly)
        self.L_w1, self.L_w2, self.L_b2 = int(L_w1), int(L_w2), int(L_b2)
        self.k1, self.k2, self.k3 = int(k1), int(k2), int(k3)
        self.t1, self.t2, self.t3 = float(tau1), float(tau2), float(tau3)

        # address bottleneck
        self.P = nn.Linear(D_in, d_addr, bias=False)
        if not learn_addr:
            for p in self.P.parameters():
                p.requires_grad = False
            with torch.no_grad():
                nn.init.normal_(self.P.weight, std=1.0/math.sqrt(D_in))

        def init_U(L, d, learn):
            U = torch.randn(L, d)
            U = U - U.mean(dim=1, keepdim=True)
            U = U / (U.norm(dim=1, keepdim=True) + 1e-8)
            return nn.Parameter(U, requires_grad=learn)

        # three unembeddings in addr-space
        self.U1 = init_U(self.L_w1, d_addr, learn_addr)
        self.U2 = init_U(self.L_w2, d_addr, learn_addr)
        self.U3 = init_U(self.L_b2, d_addr, learn_addr)

        # value tapes
        self.W1 = nn.Parameter(F.normalize(torch.randn(self.L_w1, H, D_in), dim=(1,2)), requires_grad=learn_tape_w1)
        self.W2 = nn.Parameter(F.normalize(torch.randn(self.L_w2, D_out, H), dim=(1,2)), requires_grad=learn_tape_w2)
        self.b2 = nn.Parameter(F.normalize(torch.randn(self.L_b2, D_out), dim=1),      requires_grad=learn_tape_b2)

        self.act = nn.GELU()

    @staticmethod
    def _tk(z: torch.Tensor, k: int, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
        topv, topi = torch.topk(z, k, dim=1, largest=True, sorted=False)
        w = torch.softmax(topv / (tau + 1e-8), dim=1)
        return topi, w

    def _address(self, x_addr: torch.Tensor):
        # fused logits for all three heads
        U_pack = torch.cat([self.U1, self.U2, self.U3], dim=0)          # [Ltot, d]
        Z = x_addr @ U_pack.t()                                         # [N, Ltot]
        s1, s2, s3 = self.L_w1, self.L_w2, self.L_b2
        z1, z2, z3 = torch.split( Z, (s1, s2, s3), dim=1)
        i1, w1 = FastLearnedCellX3._tk(z1, self.k1, self.t1)
        i2, w2 = FastLearnedCellX3._tk(z2, self.k2, self.t2)
        i3, w3 = FastLearnedCellX3._tk(z3, self.k3, self.t3)
        return (i1, w1), (i2, w2), (i3, w3)

    @staticmethod
    def _apply_mixture(x_flat, topi, weights, W):
        """
        Mix-then-apply (avoids replicating x):
        x_flat : [N, in]
        topi   : [N, k]
        weights: [N, k]
        W      : [L, out, in]
        return : [N, out]
        """
        N, k = topi.shape
        out_dim, in_dim = W.shape[1], W.shape[2]
        # gather -> [N, k, out, in]
        W_sel = W.index_select(0, topi.reshape(-1)).view(N, k, out_dim, in_dim)
        # mix neighbors -> [N, out, in]
        W_eff = (W_sel * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        # apply -> [N, out]
        return torch.bmm(W_eff, x_flat.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        # default values for TorchScript
        B = 1   # dummy placeholder
        T = 1
    
        if x.ndim == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
        else:
            B, T = x.shape[0], 1  # treat as sequence length 1
            x_flat = x
        
        x_addr = self.P(x_flat)                                    # [N, d_addr]
        (i1, w1), (i2, w2), (i3, w3) = self._address(x_addr)

        # FC1
        h = self._apply_mixture(x_flat, i1, w1, self.W1)           # [N, H]
        h = self.act(h)

        # FC2 + bias
        y = self._apply_mixture(h, i2, w2, self.W2)                # [N, D_out]
        b = (self.b2.index_select(0, i3.reshape(-1))
                    .view(x_flat.size(0), self.k3, self.D_out))
        y = y + (w3.unsqueeze(-1) * b).sum(dim=1)
    
        # Now both branches have valid B and T
        return y.view(B, T, self.D_out) if x.ndim == 3 else y