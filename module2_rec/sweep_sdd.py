#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量扫描 SDD，只保存 montage。
支持固定显示窗宽和固定 FOV，以便公平比较不同 SDD 下的重建结果。

说明：
- 保持与当前 sweep_sod 脚本同样的结构和调用方式。
- 不修改 fdk_recon.py。
- 继续沿用当前项目中的导入方式：from fdk_recon_sweep import ...
"""

import os
import json
import argparse
from types import SimpleNamespace

from fdk_recon_sweep import collect_projection_files, load_projections, run_reconstruction, build_arg_parser


def build_sweep_parser():
    parent = build_arg_parser()
    ap = argparse.ArgumentParser(
        description="Sweep SDD and save montage only",
        parents=[parent],
        add_help=True,
        conflict_handler="resolve",
    )
    ap.add_argument("--sdd_start", type=float, required=True)
    ap.add_argument("--sdd_end", type=float, required=True)
    ap.add_argument("--sdd_step", type=float, required=True)
    ap.add_argument("--write_reference_json", type=str, default=None, help="将首次重建得到的 window/bounds 写入 JSON，便于后续复用")
    return ap


def frange(start: float, end: float, step: float):
    if step <= 0:
        raise ValueError("sdd_step must be > 0")
    vals = []
    x = start
    eps = step * 1e-6
    while x <= end + eps:
        vals.append(round(x, 10))
        x += step
    return vals


def clone_args(ns, **updates):
    data = vars(ns).copy()
    data.update(updates)
    return SimpleNamespace(**data)


def main():
    args = build_sweep_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_projection_files(args.proj_glob, args.drop_last_if_361)
    _, projs, angles_deg, angles_rad = load_projections(args, files=files)
    preloaded = {
        "files": files,
        "projs": projs,
        "angles_deg": angles_deg,
        "angles_rad": angles_rad,
    }

    sdd_values = frange(args.sdd_start, args.sdd_end, args.sdd_step)
    base_args = clone_args(args, no_save_volume=True, montage_only=True)

    reference = {}
    if args.window_mode == "fixed" and (args.window_vmin is None or args.window_vmax is None):
        ref_sdd = sdd_values[0]
        ref_args = clone_args(base_args, sdd=ref_sdd)
        ref_result = run_reconstruction(ref_args, preloaded=preloaded, custom_prefix=f"reference_sdd_{ref_sdd:g}")
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
        ref_sdd = sdd_values[0]
        ref_args = clone_args(base_args, sdd=ref_sdd)
        ref_result = run_reconstruction(ref_args, preloaded=preloaded, custom_prefix=f"reference_fov_sdd_{ref_sdd:g}")
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
    print(f"  SDD range: {sdd_values[0]} -> {sdd_values[-1]} step={args.sdd_step}")
    print(f"  window_mode={base_args.window_mode}, vmin={base_args.window_vmin}, vmax={base_args.window_vmax}")
    print(
        "  fixed bounds="
        f"x[{base_args.x_min}, {base_args.x_max}] "
        f"y[{base_args.y_min}, {base_args.y_max}] "
        f"z[{base_args.z_min}, {base_args.z_max}]"
    )

    for sdd in sdd_values:
        if sdd <= base_args.sod:
            raise ValueError(f"Invalid scan value: sdd({sdd}) must be > sod({base_args.sod})")
        run_args = clone_args(base_args, sdd=sdd)
        prefix = f"montage_sdd_{sdd:g}"
        result = run_reconstruction(run_args, preloaded=preloaded, custom_prefix=prefix)
        print(
            f"[DONE] sdd={sdd:g} -> {result['outputs']['montage']} | "
            f"window={result['outputs']['window']}"
        )


if __name__ == "__main__":
    main()
