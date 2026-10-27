"""Diagnostic (development only): where the LTA A100 step time goes (device clock of block 0, thread 0)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nscc_common as NC  # noqa: E402,F401
import hst_common as H, expB_common as C, lta_gpu as LG
spec = H.system_spec("lta150")
for N, launches in ((1, ((32, 1), (256, 1))), (16, ((256, 1), (64, 4))), (64, ((128, 1), (32, 4))),
                    (256, ((32, 4), (32, 8), (64, 4))), (1024, ((32, 10), (32, 15), (64, 6)))):
    nst = max(4000, 400000 // N)
    cfg = C.engine_cfg_for(spec, N, 15790, nst)
    save = np.array([nst // 2, nst])
    for L in launches:
        try:
            r = LG.run_arm_gpu(cfg, "abf", N, 15790, nst, save, launch=L)
        except Exception as e:
            print(N, L, "ERR", e); continue
        m = json.loads(r["meta"]); te = m["timers_extra"]; cyc = te["cycles"]; sp = te["force_split_cycles"]
        khz = m["device_clock_khz"]; per = lambda c: c / khz * 1e3 / nst
        print(f"N {N:5d} G,WB {L}: {r['wall_total_s'] / nst * 1e6:8.2f} us/step | force {per(cyc[1]):6.2f} "
              f"[pairs {per(sp['pairs']):6.2f} reduce {per(sp['reduce']):5.2f} noise {per(sp['noise_divergence']):5.2f} "
              f"leader {per(cyc[1] - sp['pairs'] - sp['reduce'] - sp['noise_divergence']):5.2f}] dep {per(cyc[2]):5.2f} "
              f"sync {per(cyc[3]):5.2f} move {per(cyc[4]):5.2f}", flush=True)
