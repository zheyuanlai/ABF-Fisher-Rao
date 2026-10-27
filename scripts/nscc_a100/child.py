"""Run ONE job of the NSCC A100 study in this fresh process (launched by scripts/nscc_a100/runner.py).

    python scripts/nscc_a100/child.py '<job json>' [--out-root DIR]
"""
import json
import os
import sys
import traceback

# one thread for numpy / BLAS in the driver process; the engines set their own thread counts.  Must precede imports.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    job = json.loads(sys.argv[1])
    out_root = None
    if "--out-root" in sys.argv:
        out_root = sys.argv[sys.argv.index("--out-root") + 1]
    if job["impl"] != "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import nscc_common as NC
    try:
        rec = NC.run_child(job, out_root or NC.OUT_ROOT)
    except Exception:  # noqa: BLE001
        print("CHILD_FAILED " + json.dumps(job), flush=True)
        traceback.print_exc()
        return 1
    print(NC.summary_line(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
