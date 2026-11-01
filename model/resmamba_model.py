"""
GMD-Mamba — a lightweight SegResNet-style residual U-Net with a Grouped
Multi-Directional (GMD) state-space mixer applied at the deep (short-sequence)
stages ONLY. This is the efficiency idea: keep the expensive high-resolution
stages pure-convolutional, and spend the sequence model only where the token
count is small (stages 3-4, i.e. 16^3 / 8^3 for a 128^3 input).

The model class is `ResMamba3D`; the GMD scan is implemented in `GMDMamba3D`.

Scan modes (`scan=`):
  'grouped'       : Grouped Multi-Directional (GMD) scan — OURS. Splits the C
                    channels into G groups and scans each group along a different
                    axis order (the three axes and their reverses) in a single
                    pass, using a compact Mamba of width ~C/G per group. Gives
                    multi-directional 3D context at O(C^2/G) cost instead of the
                    O(K*C^2) of K full-width directional scans.
  'grouped_gated' : GMD with a lightweight channel-gated fusion of the scan
                    output (ablation variant).
  'bidir'         : simple forward+reversed full-width Mamba (baseline).

The GMD mixer enters as a residual add-on (x + GMD(x)) and is restricted to the
two deepest encoder stages, so Mamba never processes more than 16^3 = 4096
tokens. Backbone: SegResNet-style GroupNorm pre-activation ResBlocks with
additive skips (Myronenko 2018). Mamba requires a CUDA GPU. Input dims must be
divisible by 8 (three downsamplings) — e.g. 128^3 or 96^3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba as _Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False



def gn(c, groups=8):
    # pick the largest group count in {8,4,2,1} that divides c (handles widths
    # like 28/56/112 from non-power-of-two init_filters)
    g = min(groups, c)
    while c % g != 0:
        g //= 2
    return nn.GroupNorm(num_groups=max(1, g), num_channels=c)


class ResBlock(nn.Module):
    """SegResNet-style pre-activation residual block."""
    def __init__(self, c):
        super().__init__()
        self.n1 = gn(c); self.c1 = nn.Conv3d(c, c, 3, padding=1, bias=False)
        self.n2 = gn(c); self.c2 = nn.Conv3d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        z = self.c1(F.relu(self.n1(x)))
        z = self.c2(F.relu(self.n2(z)))
        return x + z


class BiMamba3D(nn.Module):
    """Simple forward+reversed Mamba token mixer (the 'bidir' fallback)."""
    def __init__(self, c, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm not available.")
        self.norm = nn.LayerNorm(c)
        self.mamba = _Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, C, D, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        tn = self.norm(t)
        y = 0.5 * (self.mamba(tn) + torch.flip(self.mamba(torch.flip(tn, dims=[1])), dims=[1]))
        return (t + y).transpose(1, 2).reshape(B, C, D, H, W)


class GMDMamba3D(nn.Module):
    """Grouped Multi-Directional Mamba (the paper's lightweight scan).

    Split the C channels into G groups; scan each group's token sequence along a
    DIFFERENT fixed axis order (D-/H-/W-major, each optionally reversed), using a
    small per-group Mamba of width ~C/G; then concat and add residually.

    Cost vs K full-width scans: K scans each cost ~ C^2 (projections); here G
    groups cost ~ G*(C/G)^2 = C^2/G  -> multi-directional coverage for a FRACTION
    of one full-width scan. No structure tensor, no learned direction selection,
    and much lighter than scanning the full width K times.
    """
    DIRS = [((0, 1, 2), False), ((1, 0, 2), False), ((2, 0, 1), False),
            ((0, 1, 2), True),  ((1, 0, 2), True),  ((2, 0, 1), True)]

    def __init__(self, c, num_dirs=6, d_state=16, d_conv=4, expand=2,
                 gated=False, gate_reduction=8):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm not available (needed for GMDMamba3D).")
        base, rem = divmod(c, num_dirs)
        self.sizes = [base + (1 if i < rem else 0) for i in range(num_dirs)]
        self.sizes = [s for s in self.sizes if s > 0]          # handle c < num_dirs
        self.dirs = self.DIRS[:len(self.sizes)]
        self.norm = nn.LayerNorm(c)
        self.mambas = nn.ModuleList([
            _Mamba(d_model=s, d_state=d_state, d_conv=d_conv, expand=expand)
            for s in self.sizes])
        # Selective fusion gate: a per-channel gate g in (0,1), computed from the
        # scan output's channel statistics, decides how much scan output y to add
        # to the residual t  ->  out = t + g * y. g->0 recovers the pure conv
        # path (so an unhelpful scan can be suppressed rather than forced in);
        # g->1 is the ungated GMD. Cheap: 2 * c^2 / reduction params.
        self.gated = gated
        if gated:
            r = max(1, c // gate_reduction)
            self.gate = nn.Sequential(
                nn.Linear(c, r), nn.ReLU(inplace=True),
                nn.Linear(r, c), nn.Sigmoid())

    @staticmethod
    def _perm(D, H, W, order, device):
        return torch.arange(D * H * W, device=device).view(D, H, W).permute(*order).reshape(-1)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("GMDMamba3D needs a CUDA tensor (mamba_ssm kernel).")
        B, C, D, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                       # (B, L, C)
        tn = self.norm(t)
        groups = torch.split(tn, self.sizes, dim=2)
        outs = []
        for gx, mamba, (order, rev) in zip(groups, self.mambas, self.dirs):
            idx = self._perm(D, H, W, order, x.device)
            xo = gx[:, idx, :]
            if rev:
                xo = torch.flip(xo, dims=[1])
            yo = mamba(xo)
            if rev:
                yo = torch.flip(yo, dims=[1])
            inv = torch.argsort(idx)
            outs.append(yo[:, inv, :])
        y = torch.cat(outs, dim=2)                             # (B, L, C)
        if self.gated:
            g = self.gate(y.mean(dim=1))                       # (B, C) channel gate
            y = y * g.unsqueeze(1)                             # modulate scan output
        return (t + y).transpose(1, 2).reshape(B, C, D, H, W)


def make_mixer(c, scan, bidirectional=True):
    if scan == "grouped":
        return GMDMamba3D(c)
    if scan == "grouped_gated":
        return GMDMamba3D(c, gated=True)
    return BiMamba3D(c)


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.c = nn.Conv3d(cin, cout, 3, stride=2, padding=1, bias=False)
    def forward(self, x): return self.c(x)


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__(); self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
    def forward(self, x): return self.up(x)


class ResMamba3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, init_filters=32,
                 blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                 mamba_stages=(2, 3), scan="grouped", bidirectional=True,
                 deep_supervision=False):
        super().__init__()
        nstage = len(blocks_down)
        chans = [init_filters * (2 ** i) for i in range(nstage)]
        self.stem = nn.Conv3d(in_channels, chans[0], 3, padding=1, bias=False)

        self.enc = nn.ModuleList()
        self.enc_mix = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, nb in enumerate(blocks_down):
            self.enc.append(nn.Sequential(*[ResBlock(chans[i]) for _ in range(nb)]))
            self.enc_mix.append(make_mixer(chans[i], scan, bidirectional)
                                if i in mamba_stages else nn.Identity())
            if i < nstage - 1:
                self.downs.append(Down(chans[i], chans[i + 1]))

        self.ups = nn.ModuleList(); self.dec = nn.ModuleList()
        self.rev = list(range(nstage - 1))[::-1]
        for j, i in enumerate(self.rev):
            self.ups.append(Up(chans[i + 1], chans[i]))
            self.dec.append(nn.Sequential(*[ResBlock(chans[i]) for _ in range(blocks_up[j])]))

        self.head = nn.Conv3d(chans[0], num_classes, 1)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.ds = nn.ModuleList([nn.Conv3d(chans[i], num_classes, 1) for i in self.rev])

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for i in range(len(self.enc)):
            x = self.enc[i](x)
            x = self.enc_mix[i](x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        aux = []
        for j, up in enumerate(self.ups):
            x = up(x)
            x = x + skips[len(self.enc) - 2 - j]
            x = self.dec[j](x)
            if self.deep_supervision and self.training:
                aux.append(self.ds[j](x))
        out = self.head(x)
        if self.deep_supervision and self.training:
            return out, aux[::-1]   # finest-first, to match ds_weights (0.5,0.25,0.125)
        return out


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for scan in ["grouped", "grouped_gated", "bidir"]:
        m = ResMamba3D(in_channels=1, num_classes=1, init_filters=32, scan=scan).to(dev)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"scan={scan:13s} HAS_MAMBA={HAS_MAMBA} params(M)={n:.2f}")
        if dev == "cuda":
            x = torch.randn(2, 1, 128, 128, 128, device=dev)
            m.train(); y = m(x); y.float().sum().backward()
            print("   fwd+bwd OK", tuple(y.shape),
                  "peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
            torch.cuda.reset_peak_memory_stats()
