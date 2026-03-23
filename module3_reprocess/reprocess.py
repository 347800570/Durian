import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def load_ct_image(file_path):
    """
    加载CT图像（npy文件）
    """
    return np.load(file_path)

def save_image(image, output_path):
    """
    保存处理后的图像为npy文件
    """
    np.save(output_path, image)

def denoise_image(image, sigma=1.0):
    """
    使用高斯滤波进行去噪处理，减少伪影的影响
    """
    return ndimage.gaussian_filter(image, sigma=sigma)

def interpolate_image(image, mask, method='linear'):
    """
    对伪影区域进行插值修复，使用邻域值进行修复
    """
    # 假设伪影区域为负值，可以根据需要调整
    image_copy = image.copy()
    image_copy[mask] = np.nan  # 将伪影区域设置为NaN

    # 对NaN值进行插值修复
    return ndimage.generic_filter(image_copy, np.nanmean, size=3)

def extract_slices(image):
    """
    提取中线切片，XY、XZ、YZ方向的切片
    """
    z, y, x = image.shape
    slices = {
        "XY": image[z // 2, :, :],  # XY方向
        "XZ": image[:, y // 2, :],  # XZ方向
        "YZ": image[:, :, x // 2],  # YZ方向
    }
    return slices

def save_slices(slices, output_dir):
    """
    保存中线切片为图像文件
    """
    for direction, slice_img in slices.items():
        plt.figure()
        plt.imshow(slice_img, cmap='gray')
        plt.title(f'{direction} Slice')
        plt.axis('off')
        plt.savefig(f'{output_dir}/{direction}_slice.png', dpi=300)
        plt.close()

def main(input_file, output_dir):
    # 1. 加载CT图像
    ct_image = load_ct_image(input_file)
    
    # 2. 伪影去除 - 高斯滤波去噪
    denoised_image = denoise_image(ct_image, sigma=2.0)
    
    # 3. 伪影区域检测 - 以负值为伪影，进行插值修复
    mask = denoised_image < 0  # 假设伪影区域为负值
    interpolated_image = interpolate_image(denoised_image, mask)
    
    # 4. 提取并保存切片
    slices = extract_slices(interpolated_image)
    save_slices(slices, output_dir)
    
    # 5. 保存去伪影后的CT图像
    save_image(interpolated_image, f'{output_dir}/processed_ct_image.npy')
    print("Processing complete. Images saved to:", output_dir)

# 使用示例
input_file = '/root/autodl-tmp/Durian/2026-3-17/output/recon_1024.npy'  # 输入的npy文件路径
output_dir = '/root/autodl-tmp/Durian/2026-3-17/output'  # 输出文件夹

main(input_file, output_dir)
