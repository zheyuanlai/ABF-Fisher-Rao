# Equal-budget ladders: implementation validation of the simulation engines

*2026-10-10.*

The engines were implemented, adversarially reviewed and fixed through a workflow:
* one implementer per engine;
* three independent reviewers per engine: spec equivalence, RNG / checkpoint / accounting, and diagnostics
  inertness;
* one fixer.

Specifications came from the Phase 0A audits (`docs/equal_budget/audit/AUDIT_0A_{gateway,lta,inventory}.md`). No
production run was launched before the engines passed.

## 1. `src/gateway_ladder_numba.py` (38 tests, `tests/test_gateway_ladder_numba.py`)

**Equivalence**
* **Q1.** The ABF arm is **bitwise** equal to `gateway_numba.run_ladder_point` / `simulate`. The test covers
  h 4e-4 and 2.5e-5 and N = 1, 16, 2048, comparing the accumulators at every save and the final state.
* **Q2.** The FR arm reproduces gateway_numba's FR law: the death/clone probabilities 1 − e^{∓gSΔt}, the cap
  max(1, ⌊0.08N⌋) on candidates, and the pool law. With the test-only switch `_fr_internal_rng=True`, which draws
  the FR uniforms from the Langevin stream as gateway_numba does, a single-FR-arm run is **bitwise** equal to
  gateway_numba. The KDE and score are bitwise equal to `kde_density` / `uniform_scores`.

**Deliberate difference: random streams**
* Langevin noise comes from numba's MT19937, seeded with gateway_numba's formula.
* FR uniforms come from a separate PCG64 stream, in fixed per-opportunity blocks.
* So the ABF and FR arms of one (N, seed), run in separate processes, see **identical Langevin noise**. Test Q4:
  with γ = 0 the FR arm is bitwise the ABF arm. The historical engine drew FR uniforms from the noise stream, so
  its arms were not separable.

**Checkpointing.** Resume is bitwise for arbitrary cut points, including FR steps (Q3).
* numba's MT19937 state is saved and restored through the private `_helperlib` API, and the struct layout is
  checked at first use.
* The FR generator state is saved via `bit_generator.state`.
* Writes are atomic.
* `run_job` refuses a result or checkpoint with a different job signature, or engine hash unless allowed.

**Diagnostics are inert.** Outputs are identical with diagnostics on or off and with different save grids (Q4).

**Walker traces.** Persistent ids follow walkers through FR gathers, so traces are trajectories, not slots.
`traces_rebirth` counts replacements.

**Accounting (Q5).** n_force_evals = N·n_steps (one evaluation per walker per step). N = 1 runs ABF only.

**Speed.** 76–99 ns per walker-step for both arms at any N.

**Latent historical edge case.** `gateway_numba.simulate` with `ess_window ≤ 0` leaves its ancestor labels
uninitialised, and its save block then indexes garbage. Production always used 4000. The new engine initialises
the labels unconditionally.

## 2. `src/lta_ladder_numba.py` (40 tests, `tests/test_lta_ladder_numba.py`)

**Equivalence**
* **E1, replay.** Driven by the torch engine's own recorded random draws (CPU float64, T 300 and 150, N 64 with
  cap 1, plus a window-start case), the discrete outputs are identical to `core_lta.run_sampler`: event counts,
  death/source indices, ancestors, transitions, crossings, histograms. Floats agree within 1e-9 Å in positions
  and 1e-10 kJ/mol in the PMF.
* **Arithmetic differences.** The LJ sum is a vectorised reassociated reduction. Bitwise reproducibility is
  therefore per compile target, which is recorded in meta and in the checkpoint signature.
* **E2.** Law tests of the engine's own FR randomness: firing frequency, sources ∝ S⁻, uniform cap subset.
* **E3.** Bitwise checkpoint/resume at N = 1, 2, 3 and 1024, including inside burn-in, at FR steps, with changed
  chunk sizes and with a zero-move final chunk.
* **E4.** The FR arm before fr_start, or with rate 0, is bitwise the ABF arm (shared noise). N = 1 runs ABF only.
* **E5.** Diagnostics are inert.
* **E6.** n_force_evals = N(n_steps + 1). The last state is also deposited, as in torch.
* **X9a–d.** Scripted event, lineage and genealogy scenarios check the diagnostic numbers themselves, not just
  their presence.

**Randomness.** Three PCG64 streams come from `SeedSequence([seed, N, round(T)])`: initial conditions (the law
of `LTASystem.initial_conditions`), Langevin noise, and FR.

**Finite-N cap.** cap(N) = max(cap_min, int(0.02 N)), with cap_min = 1 for the ladder and recorded per run.

**Statistical comparison with the CUDA production** (full knobs, N 1024, 300k steps):
* The paired FR-vs-ABF ΔI_F lies inside the published CIs at both T: −12.3 % at 300 K, −28.9 % at 150 K.
* Absolute I_F equivalence at ±5 % is not demonstrable with the 16 CUDA seeds per arm (32 numba seeds per arm;
  per-seed scatter 8–9 %). Data: `results/equal_budget_v2/equivalence/`.
* A residual ~1 % CPU-vs-CUDA difference in window occupancy is not caused by the engine: torch's own CPU random
  draws replayed through numba agree with numba's own draws.

**Speed.** 0.86–0.91 µs per molecule-step. A full arm (3.072 × 10⁸ molecule-steps) takes about 5 min of one
core at any N.

**Production smoke test, both T.** Stable under the live ABF bias:
* finite states;
* max |Γ| 29 / 21 kJ/mol/rad at N = 1024 (31.2 for the N = 1 ABF run at 300 K) against the clip 60;
* all bins visited;
* exact accounting.

## 3. Findings fixed during review (summary)

**LTA**
* The bitwise claim depended on the compile target; it is now recorded and guarded.
* The deposit clip was a free knob; it is now tied to 8 × the bias clip.
* Provenance: engine hash, git state and compile target are now in meta and in the checkpoint signature.
* `peak_rss_mb` reported the worker's lifetime peak; it is now per run.
* `diagnostics=False` left sentinel values; those keys are now omitted.
* The N = 2 FR degeneracy is flagged.

**Gateway**
* The events-per-opportunity histogram was stored only per run; it is now cumulative per save.
* FR traces followed slots instead of walkers; they now use persistent walker ids.
* `run_job` could return another job's result; signatures now prevent it.
* The default save grid lacked the physical checkpoints.
* Realised deaths are now kept separately from capped candidate counts.
* Genealogy-window age is now recorded.
* Input validation was added.
* `peak_rss_mb` is now per run.

## 4. Analysis pipeline

The analysis pipeline is described in `FINAL_RESULTS.md` §0: metrics library, per-configuration figures A–F,
synthesis S1–S8 and the completeness audit, built and adversarially reviewed with its own test suite.
