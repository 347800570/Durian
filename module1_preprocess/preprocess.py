import numpy as np
import cv2
import imageio.v2 as imageio
import glob
import os
import csv

# =========================
# 作用：基本包含从原始图像到重建输入的所有预处理步骤，包含暗场/平场校正、对数变换、滤波、增强等，可以通过滑条控制每个步骤的参数和开关，在窗口实时预览效果，最后保存为npy和bmp，并记录参数到csv。
# 使用：配置输入路径和输出路径即可。
# =========================


# =================== 路径配置 ===================
# 暗场、平场和投影文件的文件夹路径
dark_path = r"D:\Experiment\Durian\7_Gridrec\data\001\dark\*.bmp"
flat_path = r"D:\Experiment\Durian\7_Gridrec\data\001\flat\*.bmp"
proj_folder = r"D:\Experiment\Durian\7_Gridrec\data\001\durian"
# 输出路径
npy_out_folder = r"D:\Experiment\Durian\7_Gridrec\data\001\preprocess\npy"
bmp_out_folder = r"D:\Experiment\Durian\7_Gridrec\data\001\preprocess\bmp"
csv_out_folder = r"D:\Experiment\Durian\7_Gridrec\data\001\preprocess"

os.makedirs(npy_out_folder, exist_ok=True)
os.makedirs(bmp_out_folder, exist_ok=True)
os.makedirs(csv_out_folder, exist_ok=True)
EPS = 1e-6  # 防止除零或 log(0)
MAX_DISPLAY_WIDTH = 1200  # 显示窗口最大宽度
MAX_DISPLAY_HEIGHT = 800  # 显示窗口最大高度
SAVE_BMP = True  # 是否保存可视化 BMP 文件

# =================== 开关配置 ===================
# 控制每个处理步骤是否启用（滑条会控制这些开关）
steps = {
    "dark_flat": True,       # 暗场/平场校正
    "log": True,             # 对数变换
    "median": True,          # 中值滤波
    "bilateral": True,      # 双边滤波
    "clahe": True,           # CLAHE 局部对比度增强
    "gamma": True,           # 全局 gamma 校正
    "adaptive_gamma": False,  # 自适应 gamma 校正
    "sharpen": True,        # 锐化
    "segment": False,         # 分段增强果肉/果核/果皮
    "gaussian": True        # 高斯滤波
}

# =================== 参数默认值 ===================
params = {
    "median_size": 5,             # 中值滤波窗口大小

    "bilateral_d": 5,             # 双边滤波直径
    "bilateral_sigma_color": 5,   # 双边滤波颜色空间标准差
    "bilateral_sigma_space": 5,   # 双边滤波空间标准差

    "gaussian_ksize": 5,          # 高斯滤波核大小
    "gaussian_sigma": 1.0,        # 高斯滤波标准差

    "gamma": 2.0,                 # 全局 gamma

    "gamma_low": 0.3,             # 自适应 gamma 低阈值
    "gamma_high": 0.7,            # 自适应 gamma 高阈值
    "adaptive_gamma_dark": 0.8,   # 自适应 gamma 暗部
    "adaptive_gamma_bright": 1.2, # 自适应 gamma 亮部

    "clahe_clip": 2.0,            # CLAHE clip限制
    "clahe_grid": 8,              # CLAHE tileGridSize

    "sharpen_center": 5,          # 锐化卷积中心值

    "seg_low1": 0.1, "seg_high1": 0.3,  # 分段增强果肉
    "seg_low2": 0.3, "seg_high2": 0.6,  # 分段增强果核
    "seg_low3": 0.6                   # 分段增强果皮
}

# =================== 工具函数 ===================

def imread_gray(path):
    """
    读取 BMP 图像并转换为灰度 float32
    """
    img = imageio.imread(path)
    if img.ndim == 3:  # 如果是彩色图，取第一通道
        img = img[...,0]
    return img.astype(np.float32)

def load_and_average_gray(pattern):
    """
    根据路径模式读取所有暗场或平场图片，求平均
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    imgs = [imread_gray(f) for f in files]
    return np.mean(imgs, axis=0)

def robust_float_to_uint8(vol, p_lo=0.5, p_hi=99.5):
    """
    将 float32 图像映射到 uint8，用于显示或 BMP 保存
    通过百分位裁剪防止极值影响
    """
    lo = float(np.percentile(vol, p_lo))
    hi = float(np.percentile(vol, p_hi))
    if hi<=lo:
        lo, hi = float(vol.min()), float(vol.max())
        if hi<=lo:
            return np.zeros_like(vol, dtype=np.uint8)
    vol_clip = np.clip(vol, lo, hi)
    vol_norm = (vol_clip - lo)/(hi-lo)
    return (vol_norm*255+0.5).astype(np.uint8)

def gamma_correction(img, gamma):
    """
    全局 gamma 校正
    """
    img_norm = (img - img.min())/(img.max()-img.min()+1e-10)
    img_gamma = np.power(img_norm, gamma)
    return img_gamma*(img.max()-img.min()) + img.min()

def adaptive_gamma(img, gamma_low=0.3, gamma_high=0.7, gamma_dark=0.8, gamma_bright=1.2):
    """
    自适应 gamma 校正：
    低灰度部分增强暗部， 高灰度部分增强亮部
    """
    img_norm = (img - img.min())/(img.max()-img.min()+1e-10)
    img_out = np.where(img_norm<gamma_low, img_norm**gamma_dark,
                      np.where(img_norm>gamma_high, img_norm**gamma_bright, img_norm))
    return img_out*(img.max()-img.min()) + img.min()

def sharpen(img, center=5):
    """
    锐化卷积
    """
    kernel = np.array([[0,-1,0],[-1,center,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(img.astype(np.float32), -1, kernel)

def clahe_enhance(img, clip=2.0, grid=8):
    """
    CLAHE 局部对比度增强
    """
    P_u8 = robust_float_to_uint8(img)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    return clahe.apply(P_u8).astype(np.float32)

def segment_contrast(img, low1, high1, low2, high2, low3):
    """
    分段增强：将果肉/果核/果皮映射到不同灰度范围
    """
    vol_norm = (img - img.min())/(img.max()-img.min()+1e-10)
    P_seg = np.copy(vol_norm)
    # 果肉
    mask = (vol_norm>low1)&(vol_norm<high1)
    P_seg[mask] = (vol_norm[mask]-low1)/(high1-low1)
    # 果核
    mask = (vol_norm>low2)&(vol_norm<high2)
    P_seg[mask] = (vol_norm[mask]-low2)/(high2-low2)
    # 果皮
    mask = (vol_norm>low3)
    P_seg[mask] = (vol_norm[mask]-low3)/(1-low3)
    return P_seg*(img.max()-img.min()) + img.min()

def resize_for_display(img, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """
    缩放大图以适应屏幕显示
    """
    h, w = img.shape[:2]
    scale = min(max_width/w, max_height/h, 1.0)
    new_w, new_h = int(w*scale), int(h*scale)
    return cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_AREA)

def preprocess_image(img, dark_mean, flat_mean):
    """
    核心预处理函数，根据开关顺序处理
    """
    P = img.copy()
    if steps["dark_flat"]:  # 暗场/平场校正
        numerator = np.maximum(P - dark_mean, EPS)
        denominator = np.maximum(flat_mean - dark_mean, EPS)
        P = numerator/denominator
    if steps["log"]:        # 对数变换
        P = -np.log(P)
    if steps["median"]:
        P_u8 = robust_float_to_uint8(P)
        P = cv2.medianBlur(P_u8, params["median_size"]).astype(np.float32)
    if steps["bilateral"]:
        P = cv2.bilateralFilter(P.astype(np.float32),
                                params["bilateral_d"],
                                params["bilateral_sigma_color"],
                                params["bilateral_sigma_space"])
    if steps["gaussian"]:
        # 高斯滤波卷积核必须是奇数
        k = params["gaussian_ksize"]
        sigma = params["gaussian_sigma"]
        P = cv2.GaussianBlur(P, (k,k), sigma)

    if steps["gamma"]:
        P = gamma_correction(P, params["gamma"])

    if steps["adaptive_gamma"]:
        P = adaptive_gamma(P, 
                           gamma_low=params["gamma_low"],
                           gamma_high=params["gamma_high"],
                           gamma_dark=params["adaptive_gamma_dark"],
                           gamma_bright=params["adaptive_gamma_bright"])
        
    if steps["clahe"]:
        P = clahe_enhance(P, clip=params["clahe_clip"], grid=params["clahe_grid"])

    if steps["sharpen"]:
        P_uint8 = robust_float_to_uint8(P)
        P = sharpen(P_uint8, center=params["sharpen_center"])

    if steps["segment"]:
        P = segment_contrast(P,
                             params["seg_low1"], params["seg_high1"],
                             params["seg_low2"], params["seg_high2"],
                             params["seg_low3"])
    return P

# =================== 加载暗场/平场 ===================
dark_mean = load_and_average_gray(dark_path) if steps["dark_flat"] else 0
flat_mean = load_and_average_gray(flat_path) if steps["dark_flat"] else 1

# =================== 批量投影 ===================
proj_files = sorted(glob.glob(os.path.join(proj_folder,"*.bmp")))
if not proj_files:
    raise FileNotFoundError("No projection files found")

# 创建显示窗口
cv2.namedWindow("Projection Preview", cv2.WINDOW_NORMAL)
cv2.namedWindow("Params", cv2.WINDOW_NORMAL)
cv2.namedWindow("Switches", cv2.WINDOW_NORMAL)

# ========== 创建滑条（短标签版） ==========
cv2.createTrackbar("Median","Params",params["median_size"],15,lambda x: None)
cv2.createTrackbar("Bilat d","Params",params["bilateral_d"],20,lambda x: None)
cv2.createTrackbar("Bilat Color","Params",params["bilateral_sigma_color"],50,lambda x: None)
cv2.createTrackbar("Bilat Space","Params",params["bilateral_sigma_space"],50,lambda x: None)

cv2.createTrackbar("Gauss ksize","Params",params["gaussian_ksize"],15,lambda x: None)  # 必须是奇数
cv2.createTrackbar("Gauss sigma","Params",int(params["gaussian_sigma"]*10),50,lambda x: None)

cv2.createTrackbar("Gamma x10","Params",int(params["gamma"]*10),50,lambda x: None)
cv2.createTrackbar("Gamma_low","Params",int(params["gamma_low"]*10),50,lambda x: None)
cv2.createTrackbar("Gamma_high","Params",int(params["gamma_high"]*10),50,lambda x: None)
cv2.createTrackbar("AdaptG Dark","Params",int(params["adaptive_gamma_dark"]*10),50,lambda x: None)
cv2.createTrackbar("AdaptG Bright","Params",int(params["adaptive_gamma_bright"]*10),50,lambda x: None)

# 双边滤波参数滑条


cv2.createTrackbar("CLAHE Clip","Params",int(params["clahe_clip"]*10),80,lambda x: None)
cv2.createTrackbar("CLAHE Grid","Params",int(params["clahe_grid"]),40,lambda x: None)

cv2.createTrackbar("Sharpen","Params",params["sharpen_center"],20,lambda x: None)

cv2.createTrackbar("Med On","Switches",int(steps["median"]),1,lambda x: None)
cv2.createTrackbar("Bilat On","Switches",int(steps["bilateral"]),1,lambda x: None)
cv2.createTrackbar("Gauss On","Switches",int(steps["gaussian"]),1,lambda x: None)
cv2.createTrackbar("Gamma On","Switches",int(steps["gamma"]),1,lambda x: None)
cv2.createTrackbar("AdaptG On","Switches",int(steps["adaptive_gamma"]),1,lambda x: None)
cv2.createTrackbar("CLAHE On","Switches",int(steps["clahe"]),1,lambda x: None)
cv2.createTrackbar("Sharp On","Switches",int(steps["sharpen"]),1,lambda x: None)
cv2.createTrackbar("Seg On","Switches",int(steps["segment"]),1,lambda x: None)

current_index = 0

# =================== 主循环 ===================
while True:
    # -------- 更新参数和开关 ----------
    params["gamma"] = cv2.getTrackbarPos("Gamma x10","Params")/10.0
    params["gamma_low"] = cv2.getTrackbarPos("Gamma_low","Params")/10.0
    params["gamma_high"] = cv2.getTrackbarPos("Gamma_high","Params")/10.0
    params["adaptive_gamma_dark"] = cv2.getTrackbarPos("AdaptG Dark","Params")/10.0
    params["adaptive_gamma_bright"] = cv2.getTrackbarPos("AdaptG Bright","Params")/10.0

    params["median_size"] = max(1, cv2.getTrackbarPos("Median","Params")|1)

    params["clahe_clip"] = cv2.getTrackbarPos("CLAHE Clip","Params")/10.0
    params["clahe_grid"] = max(1, cv2.getTrackbarPos("CLAHE Grid","Params"))

    params["sharpen_center"] = cv2.getTrackbarPos("Sharpen","Params")
    # 更新双边滤波参数
    params["bilateral_d"] = max(1, cv2.getTrackbarPos("Bilat d","Params"))
    params["bilateral_sigma_color"] = max(0.1, cv2.getTrackbarPos("Bilat Color","Params"))
    params["bilateral_sigma_space"] = max(0.1, cv2.getTrackbarPos("Bilat Space","Params"))
    # 更新高斯滤波参数
    params["gaussian_ksize"] = max(1, cv2.getTrackbarPos("Gauss ksize","Params")|1)  # 保证奇数
    params["gaussian_sigma"] = max(0.1, cv2.getTrackbarPos("Gauss sigma","Params")/10.0)


    steps["gamma"] = cv2.getTrackbarPos("Gamma On","Switches")==1
    steps["adaptive_gamma"] = cv2.getTrackbarPos("AdaptG On","Switches")==1
    steps["median"] = cv2.getTrackbarPos("Med On","Switches")==1
    steps["bilateral"] = cv2.getTrackbarPos("Bilat On","Switches")==1
    steps["gaussian"] = cv2.getTrackbarPos("Gauss On","Switches")==1
    steps["clahe"] = cv2.getTrackbarPos("CLAHE On","Switches")==1
    steps["sharpen"] = cv2.getTrackbarPos("Sharp On","Switches")==1
    steps["segment"] = cv2.getTrackbarPos("Seg On","Switches")==1
    

    # 读取当前投影并预处理
    img = imread_gray(proj_files[current_index])
    P = preprocess_image(img, dark_mean, flat_mean)
    
    # 缩放显示
    P_display = resize_for_display(robust_float_to_uint8(P))
    cv2.imshow("Projection Preview", P_display)

    # -------- 键盘操作 --------
    key = cv2.waitKey(100) & 0xFF
    if cv2.getWindowProperty("Projection Preview", cv2.WND_PROP_VISIBLE)<1:
        break
    if key==ord("q"):      # 退出
        break
    elif key==ord("d"):    # 下一张
        current_index = (current_index+1)%len(proj_files)
    elif key==ord("a"):    # 上一张
        current_index = (current_index-1)%len(proj_files)
    elif key==ord("s"):    # 保存所有投影
        print("Starting to save.")
        for f in proj_files:
            img_f = imread_gray(f)
            P_f = preprocess_image(img_f, dark_mean, flat_mean)
            npy_path = os.path.join(npy_out_folder,os.path.basename(f).replace(".bmp",".npy"))
            os.makedirs(os.path.dirname(npy_path),exist_ok=True)
            np.save(npy_path,P_f.astype(np.float32))
            if SAVE_BMP:
                bmp_path = os.path.join(bmp_out_folder,os.path.basename(f))
                os.makedirs(os.path.dirname(bmp_path),exist_ok=True)
                P_u8 = robust_float_to_uint8(P_f)
                imageio.imwrite(bmp_path,P_u8)


        with open(os.path.join(csv_out_folder, "params_log.csv"), mode='w', newline='') as f:

            writer = csv.writer(f)
            # 写表头
            f.write("Parameter,Value\n")
            f.write(f"median_on,{int(steps['median'])}\n")
            f.write(f"bilateral_on,{int(steps['bilateral'])}\n")
            f.write(f"gaussian_on,{int(steps['gaussian'])}\n")
            f.write(f"clahe_on,{int(steps['clahe'])}\n")
            f.write(f"gamma_on,{int(steps['gamma'])}\n")
            f.write(f"adaptive_gamma_on,{int(steps['adaptive_gamma'])}\n")
            f.write(f"sharpen_on,{int(steps['sharpen'])}\n")
            f.write(f"segment_on,{int(steps['segment'])}\n\n")

            f.write(f"median_size,{params['median_size']}\n")
            f.write(f"bilateral_d,{params['bilateral_d']}\n")
            f.write(f"sigma_color,{params['bilateral_sigma_color']}\n")
            f.write(f"sigma_space,{params['bilateral_sigma_space']}\n")
            f.write(f"gaussian_ksize,{params['gaussian_ksize']}\n")
            f.write(f"gaussian_sigma,{params['gaussian_sigma']}\n")
            f.write(f"clahe_clip,{params['clahe_clip']}\n")
            f.write(f"gamma_val,{params['gamma']}\n")
            f.write(f"adaptive_gamma_dark,{params['adaptive_gamma_dark']}\n")
            f.write(f"adaptive_gamma_bright,{params['adaptive_gamma_bright']}\n")
            f.write(f"sharpen_center,{params['sharpen_center']}\n")
            

        print("All projections saved.")

cv2.destroyAllWindows()
