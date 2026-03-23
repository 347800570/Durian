import os
from PIL import Image, ImageOps
# =========================
# 作用：对像素进行翻转。
# 使用：配置输入路径和输出路径即可。
# =========================
input_folder = r"D:\Experiment\Durian\6_Durian_002\data\Durian_002_other"
output_folder = r"D:\Experiment\Durian\6_Durian_002\data\invert"

# 如果输出文件夹不存在就创建
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(".jpg"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    img = Image.open(input_path).convert("L")

    # 黑白反相
    img_inv = ImageOps.invert(img)

    img_inv.save(output_path)

    print(f"{filename} -> 已保存到 {output_path}")

print("全部完成")
