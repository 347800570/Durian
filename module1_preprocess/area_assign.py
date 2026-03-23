import numpy as np
import glob
import os
from skimage import io
import matplotlib.pyplot as plt

def remove_jig_fixed_region(input_folder, output_folder, jig_region, background_value=0.0):
    """
    简单的夹具去除脚本 - 通过固定绝对坐标区域赋值
    
    参数:
    input_folder: 输入图像文件夹路径
    output_folder: 输出图像文件夹路径  
    jig_region: 夹具区域的绝对坐标 (x_start, y_start, x_end, y_end)
    background_value: 背景值，默认为0.0
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.npy")))
    
    if not image_files:
        print("未找到.npy文件，尝试查找其他图像格式")
        # 尝试其他常见格式
        for ext in ['*.bmp', '*.tif', '*.tiff', '*.png']:
            files = glob.glob(os.path.join(input_folder, ext))
            if files:
                image_files = sorted(files)
                break
    
    if not image_files:
        raise FileNotFoundError(f"在 {input_folder} 中未找到任何图像文件")
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 解析区域坐标
    x_start, y_start, x_end, y_end = jig_region
    print(f"夹具区域: x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
    
    # 处理所有图像
    for i, img_path in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # 读取图像
        if img_path.endswith('.npy'):
            img = np.load(img_path)
        else:
            img = io.imread(img_path, as_gray=True)
            img = img.astype(np.float32)
        
        # 检查图像尺寸
        height, width = img.shape
        if x_end > width or y_end > height:
            print(f"警告: 区域坐标超出图像尺寸 {width}x{height}，自动调整")
            x_end = min(x_end, width)
            y_end = min(y_end, height)
        
        # 创建图像副本并处理夹具区域
        result = img.copy()
        result[y_start:y_end, x_start:x_end] = background_value
        
        # 保存结果
        output_filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, output_filename)
        
        if img_path.endswith('.npy'):
            np.save(output_path, result)
        else:
            # 保持原始格式
            if img_path.lower().endswith(('.bmp', '.tif', '.tiff')):
                # 对于这些格式，需要转换为uint8
                result_uint8 = (result * 255).astype(np.uint8)
                io.imsave(output_path, result_uint8)
            else:
                io.imsave(output_path, result)
        
        # 为前3张图生成预览对比图
        if i < 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 原图
            ax1.imshow(img, cmap='gray')
            ax1.set_title('原图')
            # 标记夹具区域
            rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start, 
                               fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)
            
            # 处理后的图
            ax2.imshow(result, cmap='gray')
            ax2.set_title('去除夹具后')
            
            preview_path = os.path.join(output_folder, f"preview_{i}.png")
            plt.savefig(preview_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"处理完成！结果保存到: {output_folder}")
    print(f"生成了前3张图的预览对比图")

# 使用示例
if __name__ == "__main__":
    # 根据您的图像尺寸设置夹具区域坐标
    # 格式: (x_start, y_start, x_end, y_end)
    
    # 示例：假设您的图像尺寸是1024x768，夹具在右侧200像素宽的区域
    jig_area = (1600, 1000, 1800, 1300)  # 从x=824到1024，整个高度
    
    remove_jig_fixed_region(
        input_folder=r"D:\Experiment\Durian\Durian_001_2026-1-30\preprocess_data",  # 您的预处理后数据文件夹
        output_folder=r"D:\Experiment\Durian\Durian_001_2026-1-30\preprocess_data\png\area",
        jig_region=jig_area,
        background_value=0.0  # 设置为0.0（黑色背景）
    )