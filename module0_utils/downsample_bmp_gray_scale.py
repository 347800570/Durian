from pathlib import Path
from PIL import Image

# =========================
# 作用：这个脚本用于将bmp图像下采样并转换为灰度，并重新保存为bmp。
# 配置：输入目录、输出目录、缩放比例、是否递归子目录
# =========================

# 输入目录
INPUT_DIR = r"D:\Experiment\Durian\5_Durian_001_2026-2-6\upgrade\data\clip_rotate\durian"

# 输出目录
OUTPUT_DIR = r"D:\Experiment\Durian\7_Gridrec\data\001\durian"

# 缩放比例（保持比例，不会变形）
SCALE = 0.5   # 默认约等于 2000 → 768

# 是否递归子目录
RECURSIVE = False

# =========================


def iter_bmp_files(input_dir: Path, recursive=False):
    patterns = ["*.bmp", "*.BMP"]
    if recursive:
        for p in patterns:
            yield from input_dir.rglob(p)
    else:
        for p in patterns:
            yield from input_dir.glob(p)


def process_one_image(src_path: Path, dst_path: Path, scale: float):
    with Image.open(src_path) as img:
        # 转灰度
        gray = img.convert("L")

        # 原始尺寸
        w, h = gray.size

        # 计算新尺寸（等比例）
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 防止变成0
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # 高质量缩小
        gray_ds = gray.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建目录
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 BMP
        gray_ds.save(dst_path, format="BMP")


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    files = list(iter_bmp_files(input_dir, recursive=RECURSIVE))

    if not files:
        raise ValueError(f"没有找到 BMP 文件: {input_dir}")

    print(f"[INFO] 找到 {len(files)} 张 BMP 图像")
    print(f"[INFO] 缩放比例: {SCALE}")
    print(f"[INFO] 输出目录: {output_dir}")

    for i, src_path in enumerate(files, start=1):
        if RECURSIVE:
            rel_path = src_path.relative_to(input_dir)
            dst_path = output_dir / rel_path
        else:
            dst_path = output_dir / src_path.name

        process_one_image(src_path, dst_path, SCALE)

        if i <= 5 or i % 100 == 0 or i == len(files):
            print(f"[INFO] 已处理 {i}/{len(files)}: {src_path.name}")

    print("[INFO] 全部完成")


if __name__ == "__main__":
    main()
