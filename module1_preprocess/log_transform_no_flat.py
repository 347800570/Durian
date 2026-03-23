#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log transform for CT projections when dark/flat correction is already done,
but projections are NOT normalized to 1 (air is not ~1).

For each projection:
  I0(theta) = percentile_high(I)  (e.g., 99.5th percentile over pixels)
  T = I / I0(theta)
  P = -ln(T)

Outputs 3 diagnostic images for a chosen preview index:
  1) raw
  2) normalized transmission (I / I0)
  3) log projection

Saves all log projections as .npy (float32) by default.
"""

import os, glob, argparse
import numpy as np

try:
    import imageio.v3 as iio
except Exception:
    raise ImportError("Please install imageio: pip install imageio")

import matplotlib.pyplot as plt


def read_f32(path: str) -> np.ndarray:
    return iio.imread(path).astype(np.float32)


def robust_show_range(img: np.ndarray):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1, 99])
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
    return vmin, vmax


def save_png(img: np.ndarray, path: str, title: str):
    vmin, vmax = robust_show_range(img)
    plt.figure()
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_glob", required=True, help='e.g. "/data/proj/*.tif"')
    ap.add_argument("--out_dir", required=True, help="output folder")
    ap.add_argument("--i0_percentile", type=float, default=99.5,
                    help="High percentile to estimate I0(theta) from pixels (e.g., 99.0~99.9)")
    ap.add_argument("--eps", type=float, default=1e-6, help="avoid log(0)")
    ap.add_argument("--clip_min", type=float, default=1e-6, help="min transmission clip before log")
    ap.add_argument("--clip_max", type=float, default=1.0, help="max transmission clip before log")
    ap.add_argument("--preview_index", type=int, default=0, help="0-based index for previews")
    ap.add_argument("--save_npy", action="store_true", help="save log projections as .npy (default on)")
    args = ap.parse_args()

    proj_paths = sorted(glob.glob(args.proj_glob))
    if not proj_paths:
        raise FileNotFoundError(f"No projections matched: {args.proj_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    for i, p in enumerate(proj_paths):
        raw = read_f32(p)

        # Estimate I0(theta) from high percentile over pixels (robust to noise vs max)
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            raise ValueError(f"All pixels are non-finite in: {p}")
        i0 = np.percentile(finite, args.i0_percentile)
        i0 = max(i0, args.eps)

        # Normalize to transmission-like [0, 1] (air ~ 1)
        trans = raw / i0
        trans = np.clip(trans, args.clip_min, args.clip_max)

        # Log transform
        logp = -np.log(trans + args.eps)

        # Save
        base = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(args.out_dir, f"{base}_log.npy"), logp.astype(np.float32))

        # Save previews for one chosen projection
        if i == args.preview_index:
            save_png(raw,  os.path.join(args.out_dir, "preview_1_raw.png"), f"Raw (I), I0~p{args.i0_percentile}={i0:.3f}")
            save_png(trans, os.path.join(args.out_dir, "preview_2_transmission.png"), "Transmission (I / I0)")
            save_png(logp, os.path.join(args.out_dir, "preview_3_log.png"), "Log projection = -ln(I / I0)")

    print("Done.")
    print(f"Estimated I0(theta) per projection via percentile: {args.i0_percentile}")
    print("Saved previews:")
    print("  preview_1_raw.png")
    print("  preview_2_transmission.png")
    print("  preview_3_log.png")
    print(f"Saved log projections (.npy) to: {args.out_dir}")


if __name__ == "__main__":
    main()
