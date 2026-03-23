#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from PIL import Image

# =========================
# 作用：这个脚本用于将jpg/png/bmp文件转换为npy文件。
# 使用：配置输入路径和输出路径即可。
# =========================
INPUT_DIR = r"D:\Experiment\Durian\7_Gridrec\data\002\enhance"
OUTPUT_DIR = r"D:\Experiment\Durian\7_Gridrec\data\002\npy"

# 支持的图片扩展名
PATTERNS = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]

def rgb_to_grayscale_float32(img: Image.Image) -> np.ndarray:
    """
    将 RGB 图转成灰度图，并输出为 float32 的 numpy 数组
    """
    gray = img.convert("L")   # 转灰度
    arr = np.array(gray, dtype=np.float32)
    return arr

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = []
    for pattern in PATTERNS:
        files.extend(glob.glob(os.path.join(INPUT_DIR, pattern)))

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"在输入目录中没有找到 JPG 文件: {INPUT_DIR}")

    print(f"找到 {len(files)} 个 JPG 文件，开始转换...")

    for i, file_path in enumerate(files, start=1):
        try:
            with Image.open(file_path) as img:
                arr = rgb_to_grayscale_float32(img)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_path = os.path.join(OUTPUT_DIR, base_name + ".npy")

            np.save(out_path, arr)

            print(f"[{i}/{len(files)}] 转换完成: {file_path} -> {out_path} | shape={arr.shape} dtype={arr.dtype}")

        except Exception as e:
            print(f"[{i}/{len(files)}] 转换失败: {file_path}")
            print(f"错误信息: {e}")

    print("全部处理完成。")

if __name__ == "__main__":
    main()
