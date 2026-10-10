import sys, numpy as np, time
sys.path.insert(0, "/home/zheyuanlai/ABF-Fisher-Rao/src")
import lta_ladder_numba as E
T, method, seed = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
out = sys.argv[4]
c = E.make_cfg(T, 1024, seed)
save = np.arange(0, c["n_steps"] + 1, 3000)
t = time.time()
r = E.run_arm(c, method, save, checkpoint_path=out + ".ckpt", checkpoint_every_s=120)
E.save_result(out, r)
print("done", T, method, seed, time.time() - t, flush=True)
