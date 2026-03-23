import os
from PIL import Image

def process_image(image_path, output_path, crop_box=None, rotate_angle=0):
    """
    处理单张图片：剪裁、旋转和转换为灰度图。
    
    - image_path: 输入图像的路径
    - output_path: 输出图像的路径
    - crop_box: 剪裁框（左，上，右，下）; 默认不剪裁
    - rotate_angle: 旋转角度; 默认不旋转
    """
    # 打开图片
    img = Image.open(image_path)

    # 剪裁图像（如果提供了剪裁框）
    if crop_box:
        img = img.crop(crop_box)
    
    # 旋转图像
    img = img.rotate(rotate_angle, expand=True)
    
    # 转换为灰度图
    img = img.convert("L")  # "L"模式表示灰度图
    
    # 保存处理后的图片
    img.save(output_path)

def process_folder(input_folder, output_folder, crop_box=None, rotate_angle=0):
    """
    处理文件夹中的所有BMP图片，进行剪裁、旋转和转换为灰度图。
    
    - input_folder: 输入文件夹路径
    - output_folder: 输出文件夹路径
    - crop_box: 剪裁框（左，上，右，下）; 默认不剪裁
    - rotate_angle: 旋转角度; 默认不旋转
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历文件夹中的所有BMP文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # 处理图片
            process_image(image_path, output_path, crop_box, rotate_angle)
            print(f"Processed and saved: {filename}")

def process_all_folders(input_folders, output_folders, crop_box=None, rotate_angle=0):
    """
    处理多个输入文件夹中的所有图片，输出到对应的输出文件夹。
    
    - input_folders: 输入文件夹的路径列表
    - output_folders: 输出文件夹的路径列表
    - crop_box: 剪裁框（左，上，右，下）; 默认不剪裁
    - rotate_angle: 旋转角度; 默认不旋转
    """
    for input_folder, output_folder in zip(input_folders, output_folders):
        process_folder(input_folder, output_folder, crop_box, rotate_angle)

# 示例：指定三个输入文件夹和三个输出文件夹
input_folders = [
    r"D:\Experiment\Durian\1_Original data\Durian_001_20260206\dark",  # 替换为第一个输入文件夹的路径
    r"D:\Experiment\Durian\1_Original data\Durian_001_20260206\durian",  # 替换为第二个输入文件夹的路径
    r"D:\Experiment\Durian\1_Original data\Durian_001_20260206\flat"   # 替换为第三个输入文件夹的路径
]

output_folders = [
    r"D:\Experiment\Durian\5_Durian_001_2026-2-6\upgrade\data\preprocess\dark",  # 替换为第一个输出文件夹的路径
    r"D:\Experiment\Durian\5_Durian_001_2026-2-6\upgrade\data\preprocess\durian",  # 替换为第二个输出文件夹的路径
    r"D:\Experiment\Durian\5_Durian_001_2026-2-6\upgrade\data\preprocess\flat"   # 替换为第三个输出文件夹的路径
]

# 设置剪裁框和旋转角度（可选）
crop_box = (850, 500, 2500, 2600)  # 剪裁框：左，上，右，下
rotate_angle = 90  # 旋转90度

# 处理所有文件夹中的图片
process_all_folders(input_folders, output_folders, crop_box, rotate_angle)
