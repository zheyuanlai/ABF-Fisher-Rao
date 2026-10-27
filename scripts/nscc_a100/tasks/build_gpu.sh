#!/bin/bash
# build both CUDA libraries on the compute node (login nodes throttle per-user CPU: ptxas takes > 10 min there)
python - <<'PY'
import sys, time; sys.path.insert(0, 'src/gpu')
import gpu_build as GB
for name, src in (("gw_cuda", "src/gpu/gw_cuda.cu"), ("lta_cuda", "src/gpu/lta_cuda.cu")):
    t = time.time()
    i = GB.build(name, src, ("src/gpu/gpu_common.cuh",))
    print("BUILD", name, i["tag"], "built" if i["built_in_this_process"] else "cached", f"{time.time() - t:.1f}s", flush=True)
PY
