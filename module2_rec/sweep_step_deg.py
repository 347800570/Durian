#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量扫描 step_deg，只保存 montage。
支持固定显示窗宽和固定 FOV，以便公平比较不同 step_deg 下的重建结果。

说明：
- 保持与当前 sweep_sdd / sweep_sod 脚本相同的结构和调用方式。
- 不修改 fdk_recon.py。
- 继续沿用当前项目中的导入方式：from fdk_recon_sweep import ...
- 由于 step_deg 会影响角度数组，不能直接复用旧的 angles_deg / angles_rad，
  因此本脚本会复用投影数据 projs，但在每次循环里按当前 step_deg 重新生成角度。
"""

import os
import json
import argparse
from types import SimpleNamespace

import numpy as np

from fdk_recon_sweep import collect_projection_files, load_projections, run_reconstruction, build_arg_parser


def build_sweep_parser():
    parent = build_arg_parser()
    ap = argparse.ArgumentParser(
        description="Sweep step_deg and save montage only",
        parents=[parent],
        add_help=True,
        conflict_handler="resolve",
    )
    ap.add_argument("--step_deg_start", type=float, required=True)
    ap.add_argument("--step_deg_end", type=float, required=True)
    ap.add_argument("--step_deg_step", type=float, required=True)
    ap.add_argument("--write_reference_json", type=str, default=None, help="将首次重建得到的 window/bounds 写入 JSON，便于后续复用")
    return ap


def frange(start: float, end: float, step: float):
    if step == 0:
        raise ValueError("step_deg_step must not be 0")
    if step > 0 and start > end:
        raise ValueError("当 step_deg_step > 0 时，必须满足 step_deg_start <= step_deg_end")
    if step < 0 and start < end:
        raise ValueError("当 step_deg_step < 0 时，必须满足 step_deg_start >= step_deg_end")

    vals = []
    x = start
    eps = abs(step) * 1e-6
    if step > 0:
        while x <= end + eps:
            vals.append(round(x, 10))
            x += step
    else:
        while x >= end - eps:
            vals.append(round(x, 10))
            x += step
    return vals


def clone_args(ns, **updates):
    data = vars(ns).copy()
    data.update(updates)
    return SimpleNamespace(**data)


def make_preloaded(files, projs, start_deg: float, step_deg: float):
    n = len(files)
    angles_deg = start_deg + step_deg * np.arange(n, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)
    return {
        "files": files,
        "projs": projs,
        "angles_deg": angles_deg,
        "angles_rad": angles_rad,
    }


def main():
    args = build_sweep_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_projection_files(args.proj_glob, args.drop_last_if_361)
    # 只利用 load_projections 预加载 projs；角度数组会在每次循环中根据 step_deg 重新生成
    _, projs, _, _ = load_projections(args, files=files)

    step_deg_values = frange(args.step_deg_start, args.step_deg_end, args.step_deg_step)
    base_args = clone_args(args, no_save_volume=True, montage_only=True)

    reference = {}
    if args.window_mode == "fixed" and (args.window_vmin is None or args.window_vmax is None):
        ref_step_deg = step_deg_values[0]
        ref_args = clone_args(base_args, step_deg=ref_step_deg)
        ref_preloaded = make_preloaded(files, projs, ref_args.start_deg, ref_args.step_deg)
        ref_result = run_reconstruction(
            ref_args,
            preloaded=ref_preloaded,
            custom_prefix=f"reference_step_deg_{ref_step_deg:g}",
        )
        reference["window"] = ref_result["outputs"]["window"]
        base_args.window_vmin = reference["window"]["vmin"]
        base_args.window_vmax = reference["window"]["vmax"]
        try:
            os.remove(ref_result["outputs"]["montage"])
        except OSError:
            pass

    if base_args.fixed_fov_cube is None and not base_args.fixed_bounds_json and all(
        getattr(base_args, k) is None for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    ):
        ref_step_deg = step_deg_values[0]
        ref_args = clone_args(base_args, step_deg=ref_step_deg)
        ref_preloaded = make_preloaded(files, projs, ref_args.start_deg, ref_args.step_deg)
        ref_result = run_reconstruction(
            ref_args,
            preloaded=ref_preloaded,
            custom_prefix=f"reference_fov_step_deg_{ref_step_deg:g}",
        )
        reference["bounds"] = ref_result["meta"]["bounds"]
        base_args.x_min = reference["bounds"]["x_min"]
        base_args.x_max = reference["bounds"]["x_max"]
        base_args.y_min = reference["bounds"]["y_min"]
        base_args.y_max = reference["bounds"]["y_max"]
        base_args.z_min = reference["bounds"]["z_min"]
        base_args.z_max = reference["bounds"]["z_max"]
        try:
            os.remove(ref_result["outputs"]["montage"])
        except OSError:
            pass

    if args.write_reference_json:
        payload = {
            "window_mode": base_args.window_mode,
            "window_vmin": base_args.window_vmin,
            "window_vmax": base_args.window_vmax,
            "bounds": {
                "x_min": base_args.x_min,
                "x_max": base_args.x_max,
                "y_min": base_args.y_min,
                "y_max": base_args.y_max,
                "z_min": base_args.z_min,
                "z_max": base_args.z_max,
            },
        }
        with open(args.write_reference_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[INFO] sweep settings:")
    print(f"  step_deg range: {step_deg_values[0]} -> {step_deg_values[-1]} step={args.step_deg_step}")
    print(f"  window_mode={base_args.window_mode}, vmin={base_args.window_vmin}, vmax={base_args.window_vmax}")
    print(
        "  fixed bounds="
        f"x[{base_args.x_min}, {base_args.x_max}] "
        f"y[{base_args.y_min}, {base_args.y_max}] "
        f"z[{base_args.z_min}, {base_args.z_max}]"
    )

    for step_deg in step_deg_values:
        run_args = clone_args(base_args, step_deg=step_deg)
        run_preloaded = make_preloaded(files, projs, run_args.start_deg, run_args.step_deg)
        prefix = f"montage_step_deg_{step_deg:g}"
        result = run_reconstruction(run_args, preloaded=run_preloaded, custom_prefix=prefix)
        print(
            f"[DONE] step_deg={step_deg:g} -> {result['outputs']['montage']} | "
            f"window={result['outputs']['window']}"
        )


if __name__ == "__main__":
    main()
