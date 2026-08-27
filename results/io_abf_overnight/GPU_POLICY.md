# GPU policy for this campaign (set by the user mid-run, 2026-08-27)

**GPU 3 only.** GPUs 0 and 1 carry another user's (`yesom`) jobs; GPU 2 belongs to
another session of this user's. Every launcher in this campaign now defaults to
`CUDA_VISIBLE_DEVICES=3`.

The bottleneck cells and the gateway had already finished (on GPUs 1, 2 and 3)
before this policy was set, so their artifacts stand. The WCA A0 screening was
running on GPUs 0/1/2 at that moment and was killed and restarted on GPU 3; no
screening record had been written, so nothing partial was kept.
