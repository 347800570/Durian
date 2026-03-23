#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量扫描 shift_left_px，只保存 montage。
支持固定显示窗宽和固定 FOV，以便公平比较不同 shift_left_px 下的重建结果。

说明：
- 复用 fdk_recon_sweep.py 中的重建接口，不修改原有 fdk_recon.py。
- 与 sweep_sod / sweep_sdd 保持相同风格。
- 由于 shift_left_px 会影响投影预处理，因此每个 shift 值都需要重新加载 projections，不能复用预加载的 projs。
"""

import os
import json
import argparse
from types import SimpleNamespace

from fdk_recon_sweep import collect_projection_files, load_projections, run_reconstruction, build_arg_parser


def build_sweep_parser():
    parent = build_arg_parser()
    ap = argparse.ArgumentParser(
        description="Sweep shift_left_px and save montage only",
        parents=[parent],
        add_help=True,
        conflict_handler="resolve",
    )
    ap.add_argument("--shift_left_px_start", type=int, required=True)
    ap.add_argument("--shift_left_px_end", type=int, required=True)
    ap.add_argument("--shift_left_px_step", type=int, required=True)
    ap.add_argument("--write_reference_json", type=str, default=None, help="将首次重建得到的 window/bounds 写入 JSON，便于后续复用")
    return ap


def irange(start: int, end: int, step: int):
    if step == 0:
        raise ValueError("shift_left_px_step must not be 0")
    vals = []
    if step > 0:
        if start > end:
            raise ValueError("当 shift_left_px_step > 0 时，必须满足 shift_left_px_start <= shift_left_px_end")
        x = start
        while x <= end:
            vals.append(int(x))
            x += step
    else:
        if start < end:
            raise ValueError("当 shift_left_px_step < 0 时，必须满足 shift_left_px_start >= shift_left_px_end")
        x = start
        while x >= end:
            vals.append(int(x))
            x += step
    return vals


def clone_args(ns, **updates):
    data = vars(ns).copy()
    data.update(updates)
    return SimpleNamespace(**data)


def make_preloaded(run_args, files):
    files, projs, angles_deg, angles_rad = load_projections(run_args, files=files)
    return {
        "files": files,
        "projs": projs,
        "angles_deg": angles_deg,
        "angles_rad": angles_rad,
    }


def cleanup_montage(result):
    try:
        os.remove(result["outputs"]["montage"])
    except OSError:
        pass


def main():
    args = build_sweep_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = collect_projection_files(args.proj_glob, args.drop_last_if_361)
    shift_values = irange(args.shift_left_px_start, args.shift_left_px_end, args.shift_left_px_step)
    base_args = clone_args(args, no_save_volume=True, montage_only=True)

    reference = {}

    if args.window_mode == "fixed" and (args.window_vmin is None or args.window_vmax is None):
        ref_shift = shift_values[0]
        ref_args = clone_args(base_args, shift_left_px=ref_shift, window_mode="percentile")
        ref_preloaded = make_preloaded(ref_args, files)
        ref_result = run_reconstruction(ref_args, preloaded=ref_preloaded, custom_prefix=f"reference_shift_left_px_{ref_shift:g}")
        reference["window"] = ref_result["outputs"]["window"]
        base_args.window_vmin = reference["window"]["vmin"]
        base_args.window_vmax = reference["window"]["vmax"]
        cleanup_montage(ref_result)

    if base_args.fixed_fov_cube is None and not base_args.fixed_bounds_json and all(
        getattr(base_args, k) is None for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    ):
        ref_shift = shift_values[0]
        ref_args = clone_args(base_args, shift_left_px=ref_shift)
        ref_preloaded = make_preloaded(ref_args, files)
        ref_result = run_reconstruction(ref_args, preloaded=ref_preloaded, custom_prefix=f"reference_fov_shift_left_px_{ref_shift:g}")
        reference["bounds"] = ref_result["meta"]["bounds"]
        base_args.x_min = reference["bounds"]["x_min"]
        base_args.x_max = reference["bounds"]["x_max"]
        base_args.y_min = reference["bounds"]["y_min"]
        base_args.y_max = reference["bounds"]["y_max"]
        base_args.z_min = reference["bounds"]["z_min"]
        base_args.z_max = reference["bounds"]["z_max"]
        cleanup_montage(ref_result)

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
    print(
        f"  shift_left_px range: {shift_values[0]} -> {shift_values[-1]} "
        f"step={args.shift_left_px_step}"
    )
    print(f"  window_mode={base_args.window_mode}, vmin={base_args.window_vmin}, vmax={base_args.window_vmax}")
    print(
        "  fixed bounds="
        f"x[{base_args.x_min}, {base_args.x_max}] "
        f"y[{base_args.y_min}, {base_args.y_max}] "
        f"z[{base_args.z_min}, {base_args.z_max}]"
    )

    for shift in shift_values:
        run_args = clone_args(base_args, shift_left_px=shift)
        preloaded = make_preloaded(run_args, files)
        prefix = f"montage_shift_left_px_{shift:g}"
        result = run_reconstruction(run_args, preloaded=preloaded, custom_prefix=prefix)
        print(
            f"[DONE] shift_left_px={shift:g} -> {result['outputs']['montage']} | "
            f"window={result['outputs']['window']}"
        )


if __name__ == "__main__":
    main()
