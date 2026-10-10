"""Open-flag discriminator: the numba dynamics (E1: equal to core_lta CPU given the same draws) driven by
torch's CPU RNG (MT19937): IC from LTASystem.initial_conditions, noise from torch.randn.  In law this is a
CPU-torch core_lta sample at numba speed.  ABF, 300 K, N 1024, 300000 steps, saves every 3000."""
import sys, time, types
import numpy as np, torch
HERE = "/tmp/claude-1008/-home-zheyuanlai-ABF-Fisher-Rao/b25e8a63-d0d8-4935-af2b-7d50d5f3ea3b/scratchpad/engines/fix_lta"
ROOT = "/home/zheyuanlai/ABF-Fisher-Rao"
sys.path.insert(0, HERE); sys.path.insert(0, ROOT + "/src")
import lta_ladder_numba_snap as E
from lta.core_lta import LTAParams, LTASystem
torch.set_num_threads(1)
T, method, seed, out = float(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
g = torch.Generator(device="cpu").manual_seed(7_000_000 + seed)
S = LTASystem(LTAParams(temperature=T), torch.device("cpu"), torch.float64, root=ROOT)

class TorchGen:
    def __init__(self, gen): self.g = gen; self.bit_generator = types.SimpleNamespace(state={})
    def standard_normal(self, shape): return torch.randn(tuple(shape), generator=self.g, dtype=torch.float64).numpy()
    def random(self, shape): return torch.rand(tuple(shape), generator=self.g, dtype=torch.float64).numpy()

orig = E.seed_streams
def seed_streams(seed_, N_, T_):
    gens, info = orig(seed_, N_, T_)
    info["torch_cpu_draws"] = True
    return [None, TorchGen(g), TorchGen(g)], info
E.seed_streams = seed_streams
E.initial_conditions = lambda gen, N_, a, L, r0=1.54: S.initial_conditions(1, N_, g).numpy().reshape(N_, 2, 3).copy()
c = E.make_cfg(T, 1024, seed, framework_npz=ROOT + "/cache/lta/framework.npz")
save = np.arange(0, c["n_steps"] + 1, 3000)
t = time.time()
r = E.run_arm(c, method, save)
E.save_result(out, r)
print("done", T, method, seed, time.time() - t, flush=True)
