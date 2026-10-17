"""DEVELOPMENT: the in-engine controller (engine.run with fr_controller=R1) reaches the same decision as the offline
replay of the monitor record of the same seed (dev_replay_rule.py), and its effective stop is applied."""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "adaptive_fr"))
from eri import engine as E  # noqa: E402
from eri.fr_controller import R1  # noqa: E402
from eri.model import Model  # noqa: E402
import dev_replay_rule as DR  # noqa: E402

rec = os.path.join(ROOT, "results/adaptive_fr_campaign/dev/dev_T300_K1024_rate2_s1200000.npz")
z = np.load(rec)
c = json.loads(str(z["cfg"]))
c = E.make_cfg(**dict({k: v for k, v in c.items() if k in E.DEFAULT_CFG and k != "cap"}, n_steps=80000, fr_controller=R1))
m = Model(c["variant"], T_K=c["T_K"], params=c["ff"])
res = E.run(c, "fr", model=m)
rep = DR.replay(rec)
print(json.dumps(dict(engine=dict(t1=float(res["fr_controller_t1"]), t_stop=float(res["fr_controller_t_stop"]),
                                  reason=str(res["fr_controller_reason"]), stop_step=int(res["fr_stop_effective_steps"])),
                      replay=dict(t1=rep.t1, t_stop=rep.t_stop, reason=rep.reason),
                      events_after_stop=int(res["sv_i"][-1, 2] - res["sv_i"][res["save_steps"] >= int(res["fr_stop_effective_steps"]), 2][0]))))
