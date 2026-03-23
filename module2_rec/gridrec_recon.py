#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 TomoPy 工具箱进行 Gridrec 3D 重建，并导出切片图像。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import tomopy
except ImportError as e:
    raise SystemExit(
        "TomoPy 未安装。\n"
        "请先安装：conda install -c conda-forge tomopy"
    ) from e


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gridrec reconstruction from per-angle .npy projection files"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入目录。每个 .npy 文件对应一个角度的一张 2D 投影图。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录。",
    )

    # 单次重建 center
    parser.add_argument(
        "--center",
        type=float,
        default=None,
        help="单次重建使用的旋转中心。默认 W/2。",
    )

    # center 扫描
    parser.add_argument(
        "--center_scan_start",
        type=float,
        default=None,
        help="center 扫描起点。设置后启用 center 扫描模式。",
    )
    parser.add_argument(
        "--center_scan_end",
        type=float,
        default=None,
        help="center 扫描终点。设置后启用 center 扫描模式。",
    )
    parser.add_argument(
        "--center_scan_step",
        type=float,
        default=1.0,
        help="center 扫描步长，默认 1.0。",
    )

    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.95,
        help="圆形 mask 比例，默认 0.95。<=0 表示不做。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="输出 recon.npy 的数据类型。",
    )
    parser.add_argument(
        "--no_half_turn",
        action="store_true",
        help="默认会把 0~2pi 的 parallel-beam 数据裁成前半圈 0~pi。"
             "加这个参数表示不裁剪，直接使用全部角度。",
    )

    # 扫描模式是否保存完整体数据
    parser.add_argument(
        "--save_scan_recon",
        action="store_true",
        help="center 扫描模式下也保存每个 center 的完整 recon.npy。"
             "默认不保存，只保存预览图。",
    )

    return parser.parse_args()


def natural_key(path_obj: Path):
    s = path_obj.name
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def load_projection_stack(input_dir: str) -> np.ndarray:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是目录: {input_path}")

    files = sorted(input_path.glob("*.npy"), key=natural_key)
    if not files:
        raise ValueError(f"目录中没有找到 .npy 文件: {input_path}")

    print(f"[INFO] 找到 {len(files)} 个投影文件")

    projections = []
    first_shape = None

    for i, f in enumerate(files):
        arr = np.load(f)

        if arr.ndim != 2:
            raise ValueError(
                f"{f.name} 不是 2D 投影图，当前 shape={arr.shape}，脚本期望每个文件是 (H, W)"
            )

        if first_shape is None:
            first_shape = arr.shape
        elif arr.shape != first_shape:
            raise ValueError(
                f"文件尺寸不一致：{f.name} 的 shape={arr.shape}，"
                f"第一张文件 shape={first_shape}"
            )

        projections.append(arr.astype(np.float32, copy=False))

        if i < 5:
            print(f"[INFO] 示例文件 {i}: {f.name}, shape={arr.shape}, dtype={arr.dtype}")

    stack = np.stack(projections, axis=0)   # (n_theta, H, W)
    return stack


def prepare_theta_and_proj(proj: np.ndarray, use_half_turn: bool = True):
    n_theta, H, W = proj.shape

    if use_half_turn:
        n_used = n_theta // 2
        if n_used < 2:
            raise ValueError("投影数量太少，裁半圈后无法重建。")
        proj_used = proj[:n_used]
        theta = np.linspace(0.0, np.pi, n_used, endpoint=False, dtype=np.float32)
    else:
        proj_used = proj
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)

    return proj_used, theta


def reconstruct_gridrec(proj: np.ndarray, theta: np.ndarray, center: float, mask_ratio: float):
    recon = tomopy.recon(
        proj,
        theta,
        center=center,
        sinogram_order=False,
        algorithm="gridrec",
    )

    if mask_ratio is not None and mask_ratio > 0:
        recon = tomopy.circ_mask(recon, axis=0, ratio=mask_ratio)

    return recon


def normalize_for_png(img2d: np.ndarray) -> np.ndarray:
    img = np.asarray(img2d, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.uint8)

    valid = img[finite]
    lo = np.percentile(valid, 1.0)
    hi = np.percentile(valid, 99.0)

    if hi <= lo:
        lo = float(valid.min())
        hi = float(valid.max())

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)


def save_png(img2d: np.ndarray, out_path: Path, title: str):
    img8 = normalize_for_png(img2d)
    plt.figure(figsize=(6, 6))
    plt.imshow(img8, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def center_to_name(center: float) -> str:
    # 383 -> center_383
    # 383.5 -> center_383p5
    s = f"{center:.6f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return f"center_{s}"


def save_previews(recon: np.ndarray, output_dir: Path, prefix: str = ""):
    if recon.ndim != 3:
        raise ValueError(f"重建结果不是 3D 数据，当前 shape={recon.shape}")

    z_mid = recon.shape[0] // 2
    y_mid = recon.shape[1] // 2
    x_mid = recon.shape[2] // 2

    xy = recon[z_mid, :, :]
    xz = recon[:, y_mid, :]
    yz = recon[:, :, x_mid]

    if prefix:
        xy_name = f"{prefix}_preview_xy.png"
        xz_name = f"{prefix}_preview_xz.png"
        yz_name = f"{prefix}_preview_yz.png"
    else:
        xy_name = "preview_xy.png"
        xz_name = "preview_xz.png"
        yz_name = "preview_yz.png"

    save_png(xy, output_dir / xy_name, f"{prefix} XY middle slice (z={z_mid})")
    # save_png(xz, output_dir / xz_name, f"{prefix} XZ middle slice (y={y_mid})")
    # save_png(yz, output_dir / yz_name, f"{prefix} YZ middle slice (x={x_mid})")


def frange(start: float, end: float, step: float):
    if step <= 0:
        raise ValueError("center_scan_step 必须大于 0")
    vals = []
    x = start
    eps = step * 1e-6
    while x <= end + eps:
        vals.append(round(x, 6))
        x += step
    return vals


def run_one_reconstruction(
    proj_used,
    theta,
    center,
    mask_ratio,
    save_dtype,
    output_dir: Path,
    save_recon: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = center_to_name(center)

    print(f"[INFO] 开始重建 center = {center:.3f}")
    recon = reconstruct_gridrec(
        proj=proj_used,
        theta=theta,
        center=center,
        mask_ratio=mask_ratio,
    )

    recon = recon.astype(save_dtype, copy=False)

    if save_recon:
        recon_path = output_dir / f"{prefix}_recon.npy"
        np.save(recon_path, recon)
        print(f"[INFO] 已保存: {recon_path}")
        print(f"[INFO] recon shape = {recon.shape}, dtype = {recon.dtype}")
    else:
        print(f"[INFO] 扫描模式未保存完整 recon.npy，仅保存预览图。")
        print(f"[INFO] recon shape = {recon.shape}, dtype = {recon.dtype}")

    save_previews(recon, output_dir, prefix=prefix)
    print(f"[INFO] 已保存预览图到: {output_dir}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 开始读取单角度投影文件...")
    proj = load_projection_stack(args.input_dir)
    n_theta, H, W = proj.shape
    print(f"[INFO] 投影堆栈 shape = {proj.shape}")

    use_half_turn = not args.no_half_turn
    proj_used, theta = prepare_theta_and_proj(proj, use_half_turn=use_half_turn)

    default_center = W / 2.0
    save_dtype = np.float32 if args.dtype == "float32" else np.float64

    print(f"[INFO] 是否裁前半圈: {use_half_turn}")
    print(f"[INFO] 用于重建的投影 shape = {proj_used.shape}")
    print(f"[INFO] 默认旋转中心 = {default_center:.3f}")
    print(f"[INFO] 角度数 = {len(theta)}")
    print(f"[INFO] mask_ratio = {args.mask_ratio}")

    scan_mode = (
        args.center_scan_start is not None or
        args.center_scan_end is not None
    )

    if scan_mode:
        if args.center_scan_start is None or args.center_scan_end is None:
            raise ValueError("使用 center 扫描时，必须同时提供 --center_scan_start 和 --center_scan_end")

        centers = frange(args.center_scan_start, args.center_scan_end, args.center_scan_step)
        print(f"[INFO] center 扫描列表: {centers}")
        print(f"[INFO] 扫描模式保存完整 recon.npy: {args.save_scan_recon}")

        for center in centers:
            run_one_reconstruction(
                proj_used=proj_used,
                theta=theta,
                center=center,
                mask_ratio=args.mask_ratio,
                save_dtype=save_dtype,
                output_dir=output_dir,
                save_recon=args.save_scan_recon,
            )
    else:
        center = float(args.center) if args.center is not None else default_center
        run_one_reconstruction(
            proj_used=proj_used,
            theta=theta,
            center=center,
            mask_ratio=args.mask_ratio,
            save_dtype=save_dtype,
            output_dir=output_dir,
            save_recon=True,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
