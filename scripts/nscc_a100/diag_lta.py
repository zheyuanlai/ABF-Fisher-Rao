"""Diagnostic (development only): GPU vs CPU coupling of the LTA engine per launch configuration."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402,F401
import hst_common as H, expB_common as C, lta_gpu as LG
import parallel.lta_parallel as LP
spec = H.system_spec("lta150")
for N, nst in ((16, 2000), (64, 2000)):
    seed = 15795
    cfg = C.engine_cfg_for(spec, N, seed, nst)
    save = np.arange(0, nst + 1, 25)
    rc = LP.run_arm_parallel(cfg, "abf", N, seed, nst, save, threads=min(16, N), noise="counter")
    for launch in ((32, 1), (64, 1), (256, 1), (16, 2), (8, 4)):
        try:
            rg = LG.run_arm_gpu(cfg, "abf", N, seed, nst, save, launch=launch)
        except Exception as e:
            print("N", N, launch, "ERR", e); continue
        same = np.all(np.asarray(rc["C_all"]) == np.asarray(rg["C_all"]), axis=1)
        dM = np.max(np.abs(np.asarray(rc["M_all"]) - np.asarray(rg["M_all"])) / np.maximum(np.abs(np.asarray(rc["M_all"])), 1.0), axis=1)
        h = int(save[np.argmin(same)]) if not same.all() else nst
        print(f"N {N} launch {launch}: coupled until {h}; dM at saves 0,25,50,100: {dM[0]:.2e} {dM[1]:.2e} {dM[2]:.2e} {dM[4]:.2e}; dq_final {np.max(np.abs(np.asarray(rc['q_final'])-np.asarray(rg['q_final']))):.2e}", flush=True)
