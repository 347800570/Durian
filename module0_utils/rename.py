import os
# =========================
# 作用：这个脚本用于重命名文件。
# 使用：配置输入路径即可。
# =========================
folder = r"D:\Experiment\Durian\1_Original data\Durian_002_other"   # 修改为你的文件夹路径

for filename in os.listdir(folder):

    if not filename.endswith(".jpg") and not filename.endswith(".jpeg") and not filename.endswith(".JPG") and not filename.endswith(".JPEG"):
        continue

    parts = filename.split("_")

    if len(parts) >= 2:
        number = parts[1]      # 取中间编号
        new_name = number + ".jpg"

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)

        print(f"{filename} -> {new_name}")

print("重命名完成")
