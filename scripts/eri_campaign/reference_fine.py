"""Fine-grid reference (1800-bin WHAM of the 16 MC groups), a smooth Fourier representation, the deterministic P0
floor of each histogram grid, and density-route references for grids that do not divide 180.
Writes results/eri_campaign/reference/gard_tait/reference_fine.npz and floors.json."""
import json, math, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import refanalysis as ra
from eri.model import KB
kT = KB * 600.0
base = os.path.join(ROOT, "results", "eri_campaign", "reference", "gard_tait")
acc = ra.load(os.path.join(base, "MC"), groups=list(range(16)))
H = acc["C_fine"].sum(0)
p, f, _ = ra.wham(H, acc["centres"], acc["kappa"], kT)
nf = len(p)
dphi_f = 2 * math.pi / nf
xf = (np.arange(nf) + 0.5) * dphi_f - math.pi
Ff = -kT * np.log(p / dphi_f)
Ff -= Ff.mean()
def fourier(x, y, K):
    X = np.stack([np.ones_like(x)] + [np.cos(k * x) for k in range(1, K + 1)] + [np.sin(k * x) for k in range(1, K + 1)], 1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c
def feval(c, x, K):
    X = np.stack([np.ones_like(x)] + [np.cos(k * x) for k in range(1, K + 1)] + [np.sin(k * x) for k in range(1, K + 1)], 1)
    return X @ c
out = {}
for K in (16, 24, 32, 48):
    c = fourier(xf, Ff, K)
    res = Ff - feval(c, xf, K)
    out[f"fourier_K{K}_rms_resid"] = float(np.sqrt((res ** 2).mean()))
K = 32
c = fourier(xf, Ff, K)
fs = lambda x: feval(c, np.asarray(x, float), K)
floors = {}
for ng in (60, 90, 180, 360, 720):
    d = 2 * math.pi / ng
    e = -math.pi + np.arange(ng + 1) * d
    cc = -math.pi + (np.arange(ng) + 0.5) * d
    G_inf = (fs(e[1:]) - fs(e[:-1])) / d
    F_p0 = ra.histogram_pmf(G_inf)
    Fc = fs(cc) - fs(cc).mean()
    floors[ng] = float(np.sqrt(((F_p0 - Fc - (F_p0 - Fc).mean()) ** 2).mean()))
out["P0_floor_vs_smooth_point_values"] = floors
# density-route references on 360 bins (fine probabilities summed), and its MF-route counterpart where available
p360 = p.reshape(360, nf // 360).sum(1)
F360 = -kT * np.log(p360); F360 -= F360.mean()
G360 = np.gradient(fs(-math.pi + (np.arange(360) + 0.5) * 2 * math.pi / 360), 2 * math.pi / 360)
np.savez(os.path.join(base, "reference_fine.npz"), p_fine=p, F_fine=Ff, x_fine=xf, fourier_c=c, fourier_K=K,
         F360_density=F360)
print(json.dumps(out, indent=1))
print("max |F'| (smooth) %.1f kJ/mol/rad, max |F''| %.0f" % (np.abs(np.gradient(fs(xf), dphi_f)).max(), np.abs(np.gradient(np.gradient(fs(xf), dphi_f), dphi_f)).max()))
json.dump(out, open(os.path.join(base, "floors.json"), "w"), indent=1)
