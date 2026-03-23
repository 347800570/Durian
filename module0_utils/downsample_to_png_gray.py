import os
from pathlib import Path
from PIL import Image

# =========================
# 作用：这个脚本用于将jpg图像下采样并转换为灰度，并重新保存为png。
# 配置：输入目录、输出目录、自定义输出尺寸、是否递归子目录。
# =========================

# 输入目录：原始 JPG 投影图
INPUT_DIR = r"D:\Experiment\Durian\7_Gridrec\data\002\jpg"

# 输出目录：保存降采样后的灰度 PNG
OUTPUT_DIR = r"D:\Experiment\Durian\7_Gridrec\data\002\jpg_downsample"

# 输出尺寸（可自定义）
OUTPUT_SIZE = (768, 768)

# 是否递归处理子目录
RECURSIVE = False

# =========================


def iter_image_files(input_dir: Path, recursive: bool = False):
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    if recursive:
        for pattern in patterns:
            yield from input_dir.rglob(pattern)
    else:
        for pattern in patterns:
            yield from input_dir.glob(pattern)


def process_one_image(src_path: Path, dst_path: Path, output_size=(768, 768)):
    with Image.open(src_path) as img:
        # 转灰度
        gray = img.convert("L")

        # 高质量下采样
        gray_ds = gray.resize(output_size, Image.Resampling.LANCZOS)

        # 创建输出目录
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为 PNG
        gray_ds.save(dst_path, format="PNG")


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    files = list(iter_image_files(input_dir, recursive=RECURSIVE))
    if not files:
        raise ValueError(f"在目录中没有找到 JPG 文件: {input_dir}")

    print(f"[INFO] 找到 {len(files)} 张 JPG 图像")
    print(f"[INFO] 输出尺寸: {OUTPUT_SIZE}")
    print(f"[INFO] 输出目录: {output_dir}")

    for i, src_path in enumerate(files, start=1):
        if RECURSIVE:
            rel_path = src_path.relative_to(input_dir)
            dst_path = output_dir / rel_path
        else:
            dst_path = output_dir / src_path.name

        # 后缀统一改成 .png
        dst_path = dst_path.with_suffix(".png")

        process_one_image(
            src_path=src_path,
            dst_path=dst_path,
            output_size=OUTPUT_SIZE,
        )

        if i <= 5 or i % 100 == 0 or i == len(files):
            print(f"[INFO] 已处理 {i}/{len(files)}: {src_path.name}")

    print("[INFO] 全部完成")


if __name__ == "__main__":
    main()
