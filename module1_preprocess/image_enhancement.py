import os
import glob
import csv

import cv2
import numpy as np
import imageio.v2 as imageio
# =========================
# 作用：对png图像进行增强处理，有全局均衡和CLAHE两种模式，在窗口可以实时看到增强前和增强后的效果对比图，可以调节参数，最后保存为png。
# 使用：配置输入路径和输出路径即可。
# =========================


# =================== 路径配置 ===================
# 输入 PNG 目录（灰度图）
proj_folder = r"D:\Experiment\Durian\7_Gridrec\data\002\jpg_downsample"

# 输出目录
png_out_folder = r"D:\Experiment\Durian\7_Gridrec\data\002\enhance"
csv_out_folder = r"D:\Experiment\Durian\7_Gridrec\data\002\enhance"

os.makedirs(png_out_folder, exist_ok=True)
os.makedirs(csv_out_folder, exist_ok=True)

# =================== 显示配置 ===================
MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900
PREVIEW_GAP = 20  # 原图与处理后图之间的间隔

# =================== 功能开关 ===================
# mode: 0 = 全局均衡, 1 = CLAHE
steps = {
    "hist_eq_on": True,
}

# =================== 参数默认值 ===================
params = {
    "mode": 1,            # 0=Global Equalization, 1=CLAHE
    "clahe_clip": 2.0,    # CLAHE clipLimit
    "clahe_grid": 8,      # CLAHE tileGridSize
}

# =================== 工具函数 ===================

def imread_gray(path):
    """
    读取 PNG 图像并转换为灰度 uint8
    """
    img = imageio.imread(path)
    if img.ndim == 3:
        # 如果是 RGB/RGBA，转灰度
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)


def resize_for_display(img, max_width=MAX_DISPLAY_WIDTH, max_height=MAX_DISPLAY_HEIGHT):
    """
    缩放大图以适应屏幕显示
    """
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def global_equalize(img):
    """
    全局直方图均衡（输入必须是 uint8 单通道）
    """
    return cv2.equalizeHist(img)


def clahe_enhance(img, clip=2.0, grid=8):
    """
    CLAHE 局部直方图均衡（输入必须是 uint8 单通道）
    """
    grid = max(1, int(grid))
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip)), tileGridSize=(grid, grid))
    return clahe.apply(img)


def preprocess_image(img):
    """
    根据开关和模式处理图像
    """
    P = img.copy()

    if steps["hist_eq_on"]:
        if params["mode"] == 0:
            P = global_equalize(P)
        else:
            P = clahe_enhance(P, clip=params["clahe_clip"], grid=params["clahe_grid"])

    return P


def make_side_by_side(left_img, right_img, gap=20):
    """
    将原图和处理后图拼接成左右对比图
    输入要求都是灰度 uint8
    """
    left_disp = resize_for_display(left_img)
    right_disp = resize_for_display(right_img)

    # 让两边高度一致
    h = max(left_disp.shape[0], right_disp.shape[0])

    def pad_to_height(img, target_h):
        if img.shape[0] == target_h:
            return img
        pad = target_h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    left_disp = pad_to_height(left_disp, h)
    right_disp = pad_to_height(right_disp, h)

    # 转成 BGR 方便加文字和拼接
    left_bgr = cv2.cvtColor(left_disp, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_disp, cv2.COLOR_GRAY2BGR)

    # 中间留白
    spacer = np.full((h, gap, 3), 30, dtype=np.uint8)

    canvas = np.hstack([left_bgr, spacer, right_bgr])

    # 加标题
    cv2.putText(canvas, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    right_x = left_bgr.shape[1] + gap + 10
    mode_text = "Global Equalization" if params["mode"] == 0 else "CLAHE"
    on_text = "ON" if steps["hist_eq_on"] else "OFF"
    cv2.putText(canvas, f"Processed ({mode_text}, {on_text})", (right_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def save_params_log(csv_folder):
    log_path = os.path.join(csv_folder, "params_log.csv")
    with open(log_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["hist_eq_on", int(steps["hist_eq_on"])])
        writer.writerow(["mode", "global" if params["mode"] == 0 else "clahe"])
        writer.writerow(["clahe_clip", params["clahe_clip"]])
        writer.writerow(["clahe_grid", params["clahe_grid"]])

    print(f"[INFO] 参数日志已保存: {log_path}")


# =================== 批量 PNG ===================
proj_files = sorted(glob.glob(os.path.join(proj_folder, "*.png")))
if not proj_files:
    raise FileNotFoundError(f"No PNG files found in: {proj_folder}")

# =================== 创建显示窗口 ===================
cv2.namedWindow("Projection Preview", cv2.WINDOW_NORMAL)
cv2.namedWindow("Params", cv2.WINDOW_NORMAL)
cv2.namedWindow("Switches", cv2.WINDOW_NORMAL)

# =================== 创建滑条 ===================
# Params 窗口
cv2.createTrackbar("Mode (0=G,1=C)", "Params", int(params["mode"]), 1, lambda x: None)
cv2.createTrackbar("CLAHE Clip x10", "Params", int(params["clahe_clip"] * 10), 100, lambda x: None)
cv2.createTrackbar("CLAHE Grid", "Params", int(params["clahe_grid"]), 64, lambda x: None)

# Switches 窗口
cv2.createTrackbar("HistEq On", "Switches", int(steps["hist_eq_on"]), 1, lambda x: None)

current_index = 0

# =================== 主循环 ===================
while True:
    # -------- 更新参数和开关 ----------
    params["mode"] = cv2.getTrackbarPos("Mode (0=G,1=C)", "Params")
    params["clahe_clip"] = max(0.1, cv2.getTrackbarPos("CLAHE Clip x10", "Params") / 10.0)
    params["clahe_grid"] = max(1, cv2.getTrackbarPos("CLAHE Grid", "Params"))

    steps["hist_eq_on"] = cv2.getTrackbarPos("HistEq On", "Switches") == 1

    # -------- 读取当前图像并处理 ----------
    img = imread_gray(proj_files[current_index])
    P = preprocess_image(img)

    # -------- 双图对比预览 ----------
    preview = make_side_by_side(img, P, gap=PREVIEW_GAP)

    # 左下角显示文件名
    filename = os.path.basename(proj_files[current_index])
    cv2.putText(preview, filename, (10, preview.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Projection Preview", preview)

    # -------- 键盘操作 ----------
    key = cv2.waitKey(100) & 0xFF

    if cv2.getWindowProperty("Projection Preview", cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == ord("q"):      # 退出
        break
    elif key == ord("d"):    # 下一张
        current_index = (current_index + 1) % len(proj_files)
    elif key == ord("a"):    # 上一张
        current_index = (current_index - 1) % len(proj_files)
    elif key == ord("s"):    # 保存全部
        print("[INFO] Starting to save all processed PNG files...")

        for f in proj_files:
            img_f = imread_gray(f)
            P_f = preprocess_image(img_f)

            out_path = os.path.join(png_out_folder, os.path.basename(f))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imageio.imwrite(out_path, P_f.astype(np.uint8))

        save_params_log(csv_out_folder)
        print("[INFO] All processed PNG files saved.")

cv2.destroyAllWindows()
