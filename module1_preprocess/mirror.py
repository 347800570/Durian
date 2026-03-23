import os
from PIL import Image
# =========================
# 作用：补充缺失的角度，进行镜像处理。
# 使用：配置输入路径。
# =========================

# 修改为你的文件夹
folder = r"D:\Experiment\Durian\1_Original data\Durian_002_other"

for src_angle in range(11, 27):

    dst_angle = src_angle + 180

    src_name = f"{src_angle:04d}.jpg"
    dst_name = f"{dst_angle:04d}.jpg"

    src_path = os.path.join(folder, src_name)
    dst_path = os.path.join(folder, dst_name)

    if not os.path.exists(src_path):
        print(f"源文件不存在: {src_name}")
        continue

    # 读取图片
    img = Image.open(src_path)

    # 水平翻转
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 保存
    img_flip.save(dst_path)

    print(f"{src_name} -> {dst_name}")

print("补齐完成")
