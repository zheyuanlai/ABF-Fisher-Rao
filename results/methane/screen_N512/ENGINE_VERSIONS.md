# Engine build per seed (performance-only change mid-screen, at seed boundaries)

The M-SHAKE/RATTLE solver was rewritten during the screen: host-device syncs removed
(4-9 per BAOAB step), Python pair-loops replaced by precomputed einsum structure, fixed
3-iteration Newton. Gated exactly as the campaign gates performance-only changes
(cf. torch.compile / float32): all 14 fast engine+dynamics tests pass, constraint violation
5.3e-16 nm over an 8 ps trajectory against the 1e-8 gate, equipartition vs OpenMM
LangevinMiddleIntegrator |dT| = 0.51 K (old solver measured 3.44 K on the same check).

No seed mixes engines internally; the switch happens only at seed boundaries.

| seed | engine |
|---|---|
| 5000 | pre-fix (sync solver) |
| 5004 | pre-fix (sync solver) |
| 5001-5003 | sync-free solver |
| 5005-5007 | sync-free solver |

Both builds produce trajectories from the same physical model at the same integrator
gates; seeds are independent ensembles and no gate statistic pairs seeds against each
other, so the split adds a bookkeeping row, not a confound. Recorded here rather than
discovered by a reader.
