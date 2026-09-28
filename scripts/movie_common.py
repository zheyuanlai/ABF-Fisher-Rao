"""Shared frame-rendering and encoding for the presentation movies (gateway, WCA).

A movie script defines ``draw_frame((i, path)) -> i`` that reads a module-level, read-only
dict populated BEFORE the pool is forked (py3.14 defaults to forkserver, which loses module
globals, so the pool uses the fork context) and calls :func:`render_movie`.  ffmpeg comes from
``imageio-ffmpeg`` (no system ffmpeg on this machine).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import time


def frame_indices(n_snap, stride):
    idx = list(range(0, n_snap, stride))
    if idx[-1] != n_snap - 1:
        idx.append(n_snap - 1)
    return idx


def render_movie(draw_frame, n_snap, *, stride, frames_dir, workers, fps, mp4, only=None, hold_seconds=2.0,
                 gif=None, gif_fps=12, gif_width=960, keep_frames=False, root=None):
    """Draw every ``stride``-th snapshot (always including the last) into ``frames_dir`` with a
    fork-context pool, hold the final frame for ``hold_seconds``, and encode ``mp4`` (libx264,
    crf 18, faststart) and optionally a palette-optimised ``gif``.  ``only`` renders a single
    frame (layout check) and returns."""
    idx = frame_indices(n_snap, stride)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    jobs = [(i, os.path.join(frames_dir, f"frame_{k:05d}.png")) for k, i in enumerate(idx)]
    rel = (lambda p: os.path.relpath(p, root)) if root else (lambda p: p)
    if only is not None:
        draw_frame(jobs[only]); print("wrote", jobs[only][1]); return None
    t0 = time.time()
    print(f"rendering {len(jobs)} frames ({n_snap} snapshots, stride {stride}) with {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(draw_frame, jobs, chunksize=4), 1):
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)} frames, {time.time() - t0:.0f}s", flush=True)
    print(f"frames done in {time.time() - t0:.0f}s", flush=True)

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    hold = int(round(hold_seconds * fps))
    last = jobs[-1][1]
    for k in range(len(jobs), len(jobs) + hold):
        os.link(last, os.path.join(frames_dir, f"frame_{k:05d}.png"))
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(frames_dir, "frame_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", mp4], check=True)
    print(f"movie: {rel(mp4)}  ({len(jobs)} frames + {hold} hold at {fps} fps = {(len(jobs) + hold) / fps:.1f} s, "
          f"{os.path.getsize(mp4) / 1e6:.1f} MB)")
    if gif:
        pal = os.path.join(frames_dir, "palette.png")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp4, "-vf",
                        f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,palettegen", pal], check=True)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", mp4, "-i", pal, "-lavfi",
                        f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos[x];[x][1:v]paletteuse", gif], check=True)
        print(f"gif:   {rel(gif)}  ({os.path.getsize(gif) / 1e6:.1f} MB)")
    if not keep_frames:
        shutil.rmtree(frames_dir)
    return jobs
