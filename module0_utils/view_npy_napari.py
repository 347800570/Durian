import numpy as np
import napari
# =========================
# 作用：查看重建后的npy文件。
# 使用：配置输入路径。
# =========================
path = r"D:\Experiment\Durian\7_Gridrec\output\002\single\enhance\center_324_recon.npy"
vol = np.load(path).astype(np.float32)

# 为了不卡，先降采样（看效果后再去掉这一行）
# vol = vol[::2, ::2, ::2]

v = vol[np.isfinite(vol)]

# 用更激进的分位数分开“内部”和“外壳”
p20, p60, p85, p95, p995 = np.percentile(v, [20, 60, 85, 95, 99.5])

viewer = napari.Viewer(ndisplay=3)

# --- 内部层（让核桃仁出来）---
inner = viewer.add_image(
    vol,
    name="inner",
    colormap="gray",
    rendering="translucent",   # 用真正体渲染
    opacity=0.25
)
inner.contrast_limits = (float(p60), float(p95))
inner.gamma = 0.7

# --- 外壳层（让壳保持清楚）---
shell = viewer.add_image(
    vol,
    name="shell",
    colormap="gray",
    rendering="attenuated_mip",
    opacity=0.95
)
shell.contrast_limits = (float(p95), float(p995))
shell.gamma = 1.0

napari.run()
