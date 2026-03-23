# rotate_projections_inplace_style.py
import os
import glob
import numpy as np
from PIL import Image

# =========================
# 作用：这个脚本用于将npy格式的投影图顺时针旋转90°，并保存为新的npy文件。
# 你只需要改这里的路径配置
# =========================
INPUT_GLOB = r"/root/autodl-tmp/Durian_01/area/*.npy"     # 输入投影路径（通配符）
OUTPUT_NPY_DIR = r"/root/autodl-tmp/Durian_01/spin_data" # 旋转后npy输出目录
OUTPUT_PNG_DIR = r"/root/autodl-tmp/Durian_01/spin_data/PNG"  # PNG预览输出目录
PNG_PREVIEW_COUNT = 3                          # 输出前几张PNG预览
# =========================

def rotate_clockwise_90(arr: np.ndarray) -> np.ndarray:
    """
    顺时针旋转90°：
    - 2D: (H, W) -> (W, H)
    - 3D堆栈: (N, H, W) -> (N, W, H)（对每张投影旋转）
    保持dtype不变，不插值、不滤波，仅重排索引/轴。
    """
    if arr.ndim == 2:
        return np.rot90(arr, k=-1)  # clockwise 90
    if arr.ndim == 3:
        return np.rot90(arr, k=-1, axes=(1, 2))
    raise ValueError(f"Only support 2D or 3D arrays, got shape={arr.shape}")

def to_uint8_for_preview(img2d: np.ndarray) -> np.ndarray:
    """仅用于PNG预览：鲁棒拉伸到0-255，不回写npy数据。"""
    img = np.asarray(img2d)
    if img.ndim != 2:
        raise ValueError(f"PNG preview expects 2D, got shape={img.shape}")

    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)

    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No .npy files matched INPUT_GLOB: {INPUT_GLOB}")

    os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)

    print(f"Found {len(files)} npy files.")
    print(f"Saving rotated npy to: {OUTPUT_NPY_DIR}")
    print(f"Saving PNG previews to: {OUTPUT_PNG_DIR} (first {PNG_PREVIEW_COUNT})")

    for i, fp in enumerate(files):
        arr = np.load(fp)  # dtype preserved
        rot = rotate_clockwise_90(arr)  # dtype preserved, no interpolation

        # 保存旋转后的npy（保留原文件名）
        base = os.path.basename(fp)
        out_npy = os.path.join(OUTPUT_NPY_DIR, base)
        np.save(out_npy, rot, allow_pickle=False)

        # 输出前几张PNG预览（如果是3D堆栈，默认取第0张）
        if i < PNG_PREVIEW_COUNT:
            img2d = rot[0] if rot.ndim == 3 else rot
            u8 = to_uint8_for_preview(img2d)
            out_png = os.path.join(OUTPUT_PNG_DIR, os.path.splitext(base)[0] + ".png")
            Image.fromarray(u8).save(out_png)

    # 简单校验：打印第一张的dtype和形状
    test = np.load(os.path.join(OUTPUT_NPY_DIR, os.path.basename(files[0])))
    print("Done.")
    print(f"Example output dtype={test.dtype}, shape={test.shape}")

if __name__ == "__main__":
    main()
