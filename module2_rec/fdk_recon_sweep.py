#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FDK 3D Reconstruction + slice export
- 支持作为命令行脚本单次运行
- 支持作为模块被 sweep 脚本复用
- 支持固定显示窗宽
- 支持固定 FOV / 固定体重建边界
"""

import os
import glob
import json
import argparse
from types import SimpleNamespace

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
# Display / export functions
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


def resolve_display_window(vol: np.ndarray, args):
    if args.window_mode == "fixed":
        if args.window_vmin is None or args.window_vmax is None:
            raise ValueError("window_mode=fixed 时必须同时提供 --window_vmin 和 --window_vmax")
        vmin = float(args.window_vmin)
        vmax = float(args.window_vmax)
        if vmax <= vmin:
            raise ValueError(f"Invalid fixed window: vmax({vmax}) must be > vmin({vmin})")
        return vmin, vmax
    return compute_window(vol, args.p_low, args.p_high)


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
    ax1.set_title("XY (z=mid)")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(xz, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.set_title("XZ (y=mid)")
    ax2.axis("off")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(yz, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.set_title("YZ (x=mid)")
    ax3.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Reconstruction core
# -----------------------------
def collect_projection_files(proj_glob: str, drop_last_if_361: bool = False):
    files = sorted(glob.glob(proj_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {proj_glob}")
    if len(files) == 361 and drop_last_if_361:
        files = files[:360]
    return files


def load_projections(args, files=None):
    if files is None:
        files = collect_projection_files(args.proj_glob, args.drop_last_if_361)
    n = len(files)
    angles_deg = args.start_deg + args.step_deg * np.arange(n, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)
    projs = load_stack(files, bin_factor=args.bin, shift_left_px=args.shift_left_px, dtype=args.dtype)
    return files, projs, angles_deg, angles_rad


def compute_volume_bounds(args, det_rows: int, det_cols: int, du_eff: float, dv_eff: float):
    if getattr(args, "fixed_bounds_json", None):
        with open(args.fixed_bounds_json, "r", encoding="utf-8") as f:
            bounds = json.load(f)
        required = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        missing = [k for k in required if k not in bounds]
        if missing:
            raise ValueError(f"fixed_bounds_json 缺少字段: {missing}")
        return tuple(float(bounds[k]) for k in required)

    explicit_bounds = [
        args.x_min, args.x_max,
        args.y_min, args.y_max,
        args.z_min, args.z_max,
    ]
    if any(v is not None for v in explicit_bounds):
        if not all(v is not None for v in explicit_bounds):
            raise ValueError("若使用显式固定边界，必须同时提供 x/y/z 的 min/max")
        return tuple(float(v) for v in explicit_bounds)

    if args.fixed_fov_cube is not None:
        half = 0.5 * float(args.fixed_fov_cube)
        return (-half, half, -half, half, -half, half)

    m = args.sod / args.sdd
    fov_xy = det_cols * du_eff * m * args.fov_scale_xy
    fov_z = det_rows * dv_eff * m * args.fov_scale_z
    fov_cube = max(fov_xy, fov_z) if args.fov_mode == "max" else min(fov_xy, fov_z)
    half = 0.5 * fov_cube
    return (-half, half, -half, half, -half, half)


def reconstruct_volume(args, files=None, preloaded=None):
    if preloaded is None:
        files, projs, angles_deg, angles_rad = load_projections(args, files=files)
    else:
        files = preloaded["files"]
        projs = preloaded["projs"]
        angles_deg = preloaded["angles_deg"]
        angles_rad = preloaded["angles_rad"]

    det_rows, det_cols = projs.shape[1], projs.shape[2]
    du_eff = args.du * args.bin
    dv_eff = args.dv * args.bin

    import astra

    odd = args.sdd - args.sod
    if odd <= 0:
        raise ValueError(f"Invalid distances: sdd({args.sdd}) - sod({args.sod}) must be > 0")

    proj_geom = astra.create_proj_geom(
        "cone", du_eff, dv_eff, det_rows, det_cols, angles_rad, args.sod, odd
    )

    x_min, x_max, y_min, y_max, z_min, z_max = compute_volume_bounds(
        args, det_rows, det_cols, du_eff, dv_eff
    )
    vol_geom = astra.create_vol_geom(
        args.vol_size, args.vol_size, args.vol_size,
        x_min, x_max, y_min, y_max, z_min, z_max
    )

    projs_astra = np.transpose(projs, (1, 0, 2))
    proj_id = astra.data3d.create("-proj3d", proj_geom, projs_astra)
    vol_id = astra.data3d.create("-vol", vol_geom)

    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ReconstructionDataId"] = vol_id
    cfg["ProjectionDataId"] = proj_id
    cfg["option"] = {"FilterType": args.filter}
    alg_id = astra.algorithm.create(cfg)

    try:
        astra.algorithm.run(alg_id)
        vol = astra.data3d.get(vol_id).astype(np.float32, copy=False)
        vol = reorder_volume_axes(vol, args.vol_order)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)

    meta = {
        "angles_deg_start": float(angles_deg[0]) if len(angles_deg) else None,
        "angles_deg_end": float(angles_deg[-1]) if len(angles_deg) else None,
        "num_angles": int(len(angles_deg)),
        "det_rows": int(det_rows),
        "det_cols": int(det_cols),
        "du_eff": float(du_eff),
        "dv_eff": float(dv_eff),
        "odd": float(odd),
        "bounds": {
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max),
            "z_min": float(z_min), "z_max": float(z_max),
        },
    }
    return vol, meta


def extract_mid_slices(vol: np.ndarray):
    z, y, x = vol.shape
    z0, y0, x0 = z // 2, y // 2, x // 2
    return {
        "xy": vol[z0, :, :],
        "xz": vol[:, y0, :],
        "yz": vol[:, :, x0],
        "indices": {"z0": z0, "y0": y0, "x0": x0},
    }


def default_prefix(args, custom_prefix=None):
    if custom_prefix:
        return custom_prefix
    return f"recon_vs{args.vol_size}_sod{args.sod:g}_sdd{args.sdd:g}"


def save_reconstruction_outputs(vol: np.ndarray, args, custom_prefix=None):
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = default_prefix(args, custom_prefix=custom_prefix)
    vmin, vmax = resolve_display_window(vol, args)
    slices = extract_mid_slices(vol)

    outputs = {}
    if not args.no_save_volume:
        vol_path = os.path.join(args.out_dir, f"{prefix}.npy")
        np.save(vol_path, vol)
        outputs["volume"] = vol_path

    if not args.montage_only:
        z0 = slices["indices"]["z0"]
        y0 = slices["indices"]["y0"]
        x0 = slices["indices"]["x0"]
        p_xy = os.path.join(args.out_dir, f"{prefix}_slice_xy_z{z0}.png")
        p_xz = os.path.join(args.out_dir, f"{prefix}_slice_xz_y{y0}.png")
        p_yz = os.path.join(args.out_dir, f"{prefix}_slice_yz_x{x0}.png")
        save_slice(slices["xy"], p_xy, f"XY (z={z0})", vmin, vmax, args.dpi)
        save_slice(slices["xz"], p_xz, f"XZ (y={y0})", vmin, vmax, args.dpi)
        save_slice(slices["yz"], p_yz, f"YZ (x={x0})", vmin, vmax, args.dpi)
        outputs["slice_xy"] = p_xy
        outputs["slice_xz"] = p_xz
        outputs["slice_yz"] = p_yz

    montage_path = os.path.join(args.out_dir, f"{prefix}_montage.png")
    title = (
        f"{prefix} | window=[{vmin:.4g},{vmax:.4g}] | "
        f"shape={vol.shape}"
    )
    save_montage(slices["xy"], slices["xz"], slices["yz"], montage_path, vmin, vmax, args.dpi, title)
    outputs["montage"] = montage_path
    outputs["window"] = {"vmin": float(vmin), "vmax": float(vmax)}
    outputs["slice_indices"] = slices["indices"]
    return outputs


def run_reconstruction(args, files=None, preloaded=None, custom_prefix=None):
    vol, meta = reconstruct_volume(args, files=files, preloaded=preloaded)
    outputs = save_reconstruction_outputs(vol, args, custom_prefix=custom_prefix)
    return {"volume": vol, "meta": meta, "outputs": outputs}


# -----------------------------
# Argument parsing
# -----------------------------
def build_arg_parser():
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
    ap.add_argument("--fov_mode", type=str, default="max", choices=["max", "min"])
    ap.add_argument("--fixed_fov_cube", type=float, default=None, help="固定的立方体 FOV 边长；设置后不再随 sod/sdd 自动变化")
    ap.add_argument("--x_min", type=float, default=None)
    ap.add_argument("--x_max", type=float, default=None)
    ap.add_argument("--y_min", type=float, default=None)
    ap.add_argument("--y_max", type=float, default=None)
    ap.add_argument("--z_min", type=float, default=None)
    ap.add_argument("--z_max", type=float, default=None)
    ap.add_argument("--fixed_bounds_json", type=str, default=None, help="JSON 文件，需包含 x_min/x_max/y_min/y_max/z_min/z_max")
    ap.add_argument("--start_deg", type=float, default=1.0)
    ap.add_argument("--step_deg", type=float, default=1.0)
    ap.add_argument("--drop_last_if_361", action="store_true")
    ap.add_argument("--shift_left_px", type=int, default=4)
    ap.add_argument("--bin", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--filter", type=str, default="hann", choices=["hann", "shepp-logan", "ram-lak"])
    ap.add_argument("--vol_order", type=str, default="zyx", choices=["zyx", "zxy", "yzx", "yxz", "xzy", "xyz"])
    ap.add_argument("--dpi", type=int, default=180, help="导出切片图像分辨率")
    ap.add_argument("--p_low", type=float, default=0.5, help="自动窗口下百分位")
    ap.add_argument("--p_high", type=float, default=99.5, help="自动窗口上百分位")
    ap.add_argument("--window_mode", type=str, default="percentile", choices=["percentile", "fixed"], help="percentile=每次自适应；fixed=全程固定 vmin/vmax")
    ap.add_argument("--window_vmin", type=float, default=None, help="固定窗口下界")
    ap.add_argument("--window_vmax", type=float, default=None, help="固定窗口上界")
    ap.add_argument("--no_save_volume", action="store_true", help="不保存 3D 体数据 .npy")
    ap.add_argument("--montage_only", action="store_true", help="只保存 montage，不单独保存三张切片")
    ap.add_argument("--prefix", type=str, default=None, help="输出文件名前缀")
    return ap


def parse_args():
    return build_arg_parser().parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    result = run_reconstruction(args, custom_prefix=args.prefix)
    print("Done.")
    print(f"Bounds: {result['meta']['bounds']}")
    if "volume" in result["outputs"]:
        print(f"3D volume saved: {result['outputs']['volume']}")
    print(f"Montage saved: {result['outputs']['montage']}")
    if not args.montage_only:
        print(
            "Slices saved: "
            f"{result['outputs'].get('slice_xy')}, "
            f"{result['outputs'].get('slice_xz')}, "
            f"{result['outputs'].get('slice_yz')}"
        )
    print(f"Display window: {result['outputs']['window']}")


if __name__ == "__main__":
    main()
