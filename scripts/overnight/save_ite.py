"""Save rigid ITE (ITQ-3) for the eri engine: family-A channels (axis Y = c/4 + m c, Z = n a; translation-related
only), window X from the 8-ring O atoms, cage centres midway between windows on the channel axis."""
import json, os, sys
import numpy as np
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eri import framework as fwm
CIF = os.path.join(ROOT, "cache", "overnight", "literature", "candA", "raspa2_cif_ITQ_3_cif", "ITQ-3.cif")
fw = fwm.build_orthorhombic(CIF, reps=(3, 2, 2), perm=(1, 2, 0), name="ite_itq3", n_t=64, n_o=128)
box, cell = fw["box"], fw["cell"]
b, c, a = cell
J = json.load(open(os.path.join(ROOT, "cache", "eri", "ite_itq3_channels.json")))
ch = [o for o in J["channels_thr8"] if abs(o["y_axis"] - c / 4) < 0.5 and min(o["z_axis"] % a, a - o["z_axis"] % a) < 0.5]
assert len(ch) == 1, ch
ch = ch[0]
y0, z0 = c / 4, 0.0
# 8-ring of this channel: O atoms within 4.3 A of the bead window point (min image), their mean X is the window plane
wp = np.array(ch["window_point"])
O = fw["pos"][fw["types"] == 0]
d = O - np.array([wp[0], y0, z0]); d -= box * np.round(d / box)
r = np.sqrt((d * d).sum(1))
ring = d[r < 4.6]
print("ring O count", len(ring), "radial distances", np.round(np.sort(np.sqrt(ring[:, 1] ** 2 + ring[:, 2] ** 2)), 2))
xw = (wp[0] + ring[:, 0].mean()) % b
print("window plane X = %.4f (bead-scan %.2f); ring O X spread %.3f" % (xw, wp[0], ring[:, 0].std()))
cages, wins = [], []
for k in range(3):
    for m in range(2):
        for n in range(2):
            cages.append([(xw + b / 2 + k * b) % box[0], y0 + m * c, (z0 + n * a) % box[2]])
            wins.append([(xw + k * b) % box[0], y0 + m * c, (z0 + n * a) % box[2]])
fw["variant"] = "ite_itq3"
# hidden-variable diagnostic eta = cos(4 pi Z / eta_len) = cos(2 pi Z / a): +1 on the channel axes (Z = 0, a), lower
# where the molecule sits off-axis in the cage (the cage minimum is 2.75 A off-axis along crystal a)
fwm.save_generic(fw, os.path.join(ROOT, "cache", "eri", "framework_ite_itq3.npz"), cages, wins, P=b, x_window=xw,
                 eta_len=2 * a)
print("saved; P = b = %.4f, x_window = %.4f, %d cages, %d windows (family A)" % (b, xw, len(cages), len(wins)))
json.dump(dict(P=b, x_window=xw, channel_axis_YZ=[y0, z0], cages=cages, windows=wins, ring_O=len(ring),
               npz_sha256=fwm.sha256_file(os.path.join(ROOT, "cache", "eri", "framework_ite_itq3.npz")),
               source=fw["source"], source_sha256=fw["source_sha256"]),
          open(os.path.join(ROOT, "cache", "eri", "framework_ite_itq3.json"), "w"), indent=1)
