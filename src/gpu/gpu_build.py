"""Build / load the CUDA libraries of the A100 backends (src/gpu/*.cu) with nvcc; shared by gateway_gpu / lta_gpu.

BUILD: nvcc -O3 -arch=sm_80 --fmad=false -Xcompiler -fPIC -shared (cudart linked statically), output
$GPU_BUILD_DIR/<name>_<tag>.so with tag = sha256(sources, nvcc, flags, nvcc version)[:16]; a changed source or
compiler gives a new tag.  --fmad=false: no FMA contraction (the CPU engines are built without it); float64 division
and sqrt are IEEE in CUDA.  NVCC defaults to `nvcc` on PATH (module load cuda/12.8.1 on NSCC ASPIRE2A).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NVCC = os.environ.get("NVCC", "nvcc")
ARCH = os.environ.get("GPU_ARCH", "sm_80")
FLAGS = ("-O3", f"-arch={ARCH}", "--fmad=false", "-std=c++17", "-Xcompiler", "-fPIC", "-shared")
BUILD_DIR = os.environ.get("GPU_BUILD_DIR", os.path.join(ROOT, "build", "gpu"))


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def nvcc_version():
    try:
        out = subprocess.run([NVCC, "--version"], capture_output=True, text=True, timeout=60).stdout
    except OSError:
        return None
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else None


def build(name, main_src, deps=()):
    """Compile main_src (+ header deps) into a tagged .so if absent; return the build record."""
    srcs = [main_src, *deps]
    ver = nvcc_version()
    tag_in = "\n".join([*(sha256_file(p) for p in srcs), NVCC, " ".join(FLAGS), str(ver)])
    tag = hashlib.sha256(tag_in.encode()).hexdigest()[:16]
    os.makedirs(BUILD_DIR, exist_ok=True)
    so = os.path.join(BUILD_DIR, f"{name}_{tag}.so")
    built = False
    if not os.path.exists(so):
        if ver is None:
            raise RuntimeError(f"{NVCC} not found (module load cuda/12.8.1) and no prebuilt {so}")
        tmp = so + f".tmp{os.getpid()}"
        cmd = [NVCC, *FLAGS, "-o", tmp, main_src]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc build failed ({' '.join(cmd)}):\n{r.stderr}")
        os.replace(tmp, so)
        built = True
    return dict(so_path=so, so_sha256=sha256_file(so), sources_sha256={os.path.relpath(p, ROOT): sha256_file(p)
                                                                         for p in srcs},
                nvcc=NVCC, nvcc_version=ver, flags=list(FLAGS), tag=tag, built_in_this_process=built)


def load(name, main_src, deps=()):
    info = build(name, main_src, deps)
    return ctypes.CDLL(info["so_path"]), info


def ptr(a):
    return ctypes.c_void_p(a.ctypes.data)


def device_info(lib, prefix, dev=0):
    out = (ctypes.c_longlong * 8)()
    name = ctypes.create_string_buffer(256)
    fn = getattr(lib, f"{prefix}_gpu_device_info")
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    if fn(int(dev), ctypes.cast(out, ctypes.c_void_p), name, 256) != 0:
        return None
    return dict(name=name.value.decode(), sm_count=int(out[0]), clock_khz=int(out[1]), smem_optin=int(out[2]),
                cc=f"{out[3]}.{out[4]}", ecc=bool(out[5]), mem_mb=int(out[6]),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def git_state():
    def g(*a):
        try:
            return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001 (compute nodes have no git)
            return ""
    commit = g("rev-parse", "HEAD") or os.environ.get("ABFFR_GIT_COMMIT", "unknown")
    return dict(git_commit=commit, git_commit_source=("git" if g("rev-parse", "HEAD") else "env ABFFR_GIT_COMMIT"))
