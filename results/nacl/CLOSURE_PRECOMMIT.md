# NaCl study closure: what each N = 32 outcome means, written before the result exists

Committed 2026-08-15 ~01:00 UTC, while the N = 32 cell is at step 0 of 1562500 on GPU 2. Nothing
below may be revised after the cell reports; if it turns out to be wrong, it is amended by a
numbered amendment with the original left in place. The point of writing it now is that the
mapping from outcome to conclusion cannot be chosen once the number is visible.

## The state of the ladder at the moment of writing

```
N = 64   DECIDED, ABF-SUFFICIENT 8/8. Gate C SSIP lambda_min 61.45, 0/8 deficient.
         CIP unpowered (lambda_min 1.57) and cleared instead by the windowed audit:
         worst of 47 windows 1.111 +- 0.078, 2-sigma [0.954, 1.267] against a 0.5 threshold.
N = 32   RUNNING. The ONLY remaining cell that can return a verdict.
N = 16   STRUCTURALLY UNCLASSIFIABLE, a priori. Needs Q* = 1 (one basin holding the entire
N =  8   target) to reach lambda = 16. Struck without data; no sampling changes it.
```

Projected power at N = 32, from the accepted reference and nothing else: **SSIP lambda ~ 30.7
(powered), CIP lambda ~ 0.78 (unpowered)**. So SSIP is expected to carry the cell exactly as it
carried N = 64, and CIP is expected to need the windowed audit again. Recording the projection
now so that if the realised lambda differs materially, that discrepancy is itself a finding
about the bias-aware target rather than something absorbed silently.

## The four outcomes and their conclusions

**(1) SSIP powered, no sustained deficit** *(the expected outcome)*
The classifiable ladder is {32, 64} and no cell is under-established. **Study verdict: NaCl is
not an mFR candidate under the preregistered budget -- ABF-sufficient wherever the question can
be asked.** Close the study. The weak-null caveat below attaches to this and is not optional.

**(2) SSIP powered, sustained deficit on >= 6/8 seeds**
N = 32 is establishment-limited and **mFR is licensed at N = 32** -- the first NaCl cell that
would be. Then, and only then: Gate D needs tau_perp, which means re-running the tau_perp job I
stopped at 00:20 (it has no checkpointing, ~4 h on GPU 2), followed by mFR calibration and the
five preregistered arms. This is the outcome that costs the most and it must not be reached by
relaxing anything -- in particular the deficit must clear the *contiguity* requirement, not
merely dip below 0.5 Q\*.

**(3) CIP unpowered and SSIP powered-and-clear, i.e. the N = 64 shape**
Treat CIP exactly as at N = 64: report Gate C NON-BINDING there and run
`nacl_audit_cip_power.py` on the N = 32 cell. **The windowed result is the CIP claim, and it may
come back INCONCLUSIVE** -- the 2-sigma band straddling 0.5 -- in which case CIP at N = 32 is
UNKNOWN and must be reported as unknown, not folded into the null. Outcome (1) then holds only
for SSIP and the study closes as "ABF-sufficient on the state that can be measured, undetermined
on the other". That is a weaker closure than N = 64's and must be written as one.

**(4) No state at N = 32 reaches lambda >= 16**
The cell is UNCLASSIFIABLE and the classifiable ladder collapses to {64} alone. The study still
closes -- there is no smaller cell that can be asked -- but on a single cell, and the write-up
says so. **It does not license extending N = 32 until it classifies**: that is choosing the
sample size against the result.

## The caveat, pre-committed at commit `addfbed` and repeated here unchanged

NaCl's hydration varies **14-83x across r versus at fixed r**, against methane's 5.4x. NaCl
therefore has little structure orthogonal to the reaction coordinate, and **"mFR had nothing to
work with" is a live alternative to "mFR was not needed."** Whatever the outcome above, this
null is **weaker than methane's** and must not be reported as a second independent null of equal
strength. Gate A at 1.000 is a statement about NaCl's physics, not a strong gate.

## What is NOT permitted on any branch

* extending N = 32, adding seeds, or lengthening T because the answer came back unpowered or
  marginal -- the budget is preregistered and the sample size is not a free parameter;
* relaxing LAMBDA_MIN from 16 to admit a cell (N = 16 misses by 0.64 walkers, which is exactly
  the temptation the frozen threshold exists to refuse);
* reading Gate D or any mFR arm on a cell whose Gate C did not fire on a POWERED state;
* reporting outcome (3) or (4) as if it were outcome (1).
