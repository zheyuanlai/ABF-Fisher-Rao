import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_tdraw as S   # prints the table again
rng = np.random.default_rng(20261010)
def boot(a, b, B=10000):
    ia = rng.integers(0, a.size, (B, a.size)); ib = rng.integers(0, b.size, (B, b.size))
    r = np.median(a[ia], 1) / np.median(b[ib], 1)
    return np.median(a) / np.median(b), np.quantile(r, [0.05, 0.95])
MARG = dict(IF=0.05, eT=0.15, cx=0.05, fwin=0.05)
for key, mg in MARG.items():
    a = np.array([r[key] for r in S.tdr]); b = np.array([r[key] for r in S.nat]); c = np.asarray(S.prod[key], float)
    for nm, x, y in (("numba-native / torchCPU-draws", b, a), ("torchCPU-draws / prod CUDA", a, c)):
        est, (lo, hi) = boot(x, y)
        print(f"TOST {key:4s} {nm:30s}: median ratio {est:.4f} 90% CI [{lo:.4f}, {hi:.4f}] +-{mg:.0%} -> "
              f"{'EQUIVALENT' if lo > 1 - mg and hi < 1 + mg else 'NOT SHOWN'}")
