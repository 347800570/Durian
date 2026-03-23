#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 ASTRA 工具箱进行 FDK 3D 重建，并导出切片图像。
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions
# -----------------------------
def downscale_mean(img: np.ndarray, bin_factor: int) -> np.ndarray:
    if bin_factor == 1:
        return img
    h, w = img.shape
    if h % bin_factor != 0 or w % bin_factor != 0:
        raise ValueError(f"Image shape {img.shape} not divisible by bin={bin_factor}")
    img = img.reshape(h // bin_factor, bin_factor, w // bin_factor, bin_factor)
    return img.mean(axis=(1, 3))

def load_stack(file_list, bin_factor: int, shift_left_px: int, dtype: str) -> np.ndarray:
    first = np.load(file_list[0]).astype(dtype, copy=False)
    first = downscale_mean(first, bin_factor)
    if shift_left_px:
        first = np.roll(first, shift=-shift_left_px, axis=1)
    rows, cols = first.shape
    n = len(file_list)
    stack = np.empty((n, rows, cols), dtype=np.dtype(dtype))
    stack[0] = first
    for i in range(1, n):
        img = np.load(file_list[i]).astype(dtype, copy=False)
        img = downscale_mean(img, bin_factor)
        if shift_left_px:
            img = np.roll(img, shift=-shift_left_px, axis=1)
        stack[i] = img
    return stack

def reorder_volume_axes(vol: np.ndarray, order: str) -> np.ndarray:
    perm_map = {
        "zyx": (0, 1, 2),
        "zxy": (0, 2, 1),
        "yzx": (1, 0, 2),
        "yxz": (1, 2, 0),
        "xzy": (2, 0, 1),
        "xyz": (2, 1, 0),
    }
    return np.transpose(vol, perm_map[order])

# -----------------------------
# Slice export functions
# -----------------------------
def compute_window(vol: np.ndarray, p_low: float, p_high: float):
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1e-6
    return lo, hi

def save_slice(img: np.ndarray, path: str, title: str, vmin: float, vmax: float, dpi: int):
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_montage(xy, xz, yz, path, vmin, vmax, dpi, title):
    fig = plt.figure(figsize=(12, 4), dpi=dpi)
    fig.suptitle(title)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(xy, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("XY (z=mid)"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(xz, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.set_title("XZ (y=mid)"); ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(yz, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.set_title("YZ (x=mid)"); ax3.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="FDK 3D reconstruction + slice export")
    ap.add_argument("--proj_glob", type=str, required=True, help="投影文件 glob")
    ap.add_argument("--out_dir", type=str, required=True, help="输出文件夹")
    ap.add_argument("--vol_size", type=int, default=512)
    ap.add_argument("--sod", type=float, default=200.66)
    ap.add_argument("--sdd", type=float, default=553.74)
    ap.add_argument("--du", type=float, default=0.05)
    ap.add_argument("--dv", type=float, default=0.05)
    ap.add_argument("--fov_scale_xy", type=float, default=0.8)
    ap.add_argument("--fov_scale_z", type=float, default=0.8)
    ap.add_argument("--fov_mode", type=str, default="max", choices=["max","min"])
    ap.add_argument("--start_deg", type=float, default=1.0)
    ap.add_argument("--step_deg", type=float, default=1.0)
    ap.add_argument("--drop_last_if_361", action="store_true")
    ap.add_argument("--shift_left_px", type=int, default=4)
    ap.add_argument("--bin", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    ap.add_argument("--filter", type=str, default="hann", choices=["hann","shepp-logan","ram-lak"])
    ap.add_argument("--vol_order", type=str, default="zyx",
                    choices=["zyx","zxy","yzx","yxz","xzy","xyz"])
    ap.add_argument("--dpi", type=int, default=180, help="导出切片图像的分辨率")
    ap.add_argument("--p_low", type=float, default=0.5, help="显示窗口下百分位")
    ap.add_argument("--p_high", type=float, default=99.5, help="显示窗口上百分位")

    
    return ap.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    print("Starting FDK reconstruction...")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(args.proj_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.proj_glob}")
    n = len(files)
    if n == 361 and args.drop_last_if_361:
        files = files[:360]
        n = 360

    angles_deg = args.start_deg + args.step_deg * np.arange(n, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)

    # Load projections
    projs = load_stack(files, bin_factor=args.bin, shift_left_px=args.shift_left_px, dtype=args.dtype)
    det_rows, det_cols = projs.shape[1], projs.shape[2]
    du_eff = args.du * args.bin
    dv_eff = args.dv * args.bin

    import astra
    odd = args.sdd - args.sod
    if odd <= 0:
        raise ValueError(f"Invalid distances: sdd({args.sdd}) - sod({args.sod}) must be > 0")

    # Projection geometry
    proj_geom = astra.create_proj_geom('cone', du_eff, dv_eff, det_rows, det_cols, angles_rad, args.sod, odd)

    # Volume geometry
    m = args.sod / args.sdd
    fov_xy = det_cols * du_eff * m * args.fov_scale_xy
    fov_z = det_rows * dv_eff * m * args.fov_scale_z
    fov_cube = max(fov_xy, fov_z) if args.fov_mode=="max" else min(fov_xy, fov_z)
    x_min, x_max = -0.5*fov_cube, 0.5*fov_cube
    y_min, y_max = -0.5*fov_cube, 0.5*fov_cube
    z_min, z_max = -0.5*fov_cube, 0.5*fov_cube
    vol_geom = astra.create_vol_geom(args.vol_size, args.vol_size, args.vol_size,
                                     x_min, x_max, y_min, y_max, z_min, z_max)

    # ASTRA data objects
    projs_astra = np.transpose(projs, (1,0,2))
    proj_id = astra.data3d.create('-proj3d', proj_geom, projs_astra)
    vol_id = astra.data3d.create('-vol', vol_geom)

    # FDK reconstruction
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = vol_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {'FilterType': args.filter}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    vol = astra.data3d.get(vol_id).astype(np.float32, copy=False)
    vol = reorder_volume_axes(vol, args.vol_order)

    # Cleanup ASTRA objects
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)

    # Save 3D volume
    vol_path = os.path.join(args.out_dir, f"recon_{args.vol_size}.npy")
    np.save(vol_path, vol)

    # -----------------------------
    # Export slices
    # -----------------------------
    z, y, x = vol.shape
    z0, y0, x0 = z//2, y//2, x//2
    xy, xz, yz = vol[z0,:,:], vol[:,y0,:], vol[:,:,x0]
    vmin, vmax = compute_window(vol, args.p_low, args.p_high)

    prefix = f"recon_{args.vol_size}"
    p_xy = os.path.join(args.out_dir, f"{prefix}_slice_xy_z{z0}.png")
    p_xz = os.path.join(args.out_dir, f"{prefix}_slice_xz_y{y0}.png")
    p_yz = os.path.join(args.out_dir, f"{prefix}_slice_yz_x{x0}.png")
    p_m = os.path.join(args.out_dir, f"{prefix}_montage.png")

    save_montage(xy, xz, yz, p_m, vmin, vmax, args.dpi,
                 title=f"{prefix}: window=[{vmin:.4g},{vmax:.4g}] shape={vol.shape}")

    print("Done.")
    print(f"3D volume saved: {vol_path}")
    print(f"Slices saved: {p_xy}, {p_xz}, {p_yz}, {p_m}")

if __name__ == "__main__":
    main()
