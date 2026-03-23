
# =========================
# 作用：输出重建后的XY、XZ、YZ切片以及它们的拼图。
# 使用：配置输入、输出路径，还可以配置切片名字。
# =========================
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_window(vol: np.ndarray, p_low: float, p_high: float):
    """Compute robust display window from percentiles on the whole volume."""
    # use finite values only
    v = vol[np.isfinite(vol)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1e-6
    return lo, hi


def save_slice(img: np.ndarray, path: str, title: str, vmin: float, vmax: float, dpi: int):
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_montage(xy: np.ndarray, xz: np.ndarray, yz: np.ndarray,
                 path: str, vmin: float, vmax: float, dpi: int, title: str):
    fig = plt.figure(figsize=(12, 4), dpi=dpi)
    fig.suptitle(title)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(xy, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("XY (z=mid)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(xz, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.set_title("XZ (y=mid)")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(yz, cmap="gray", vmin=vmin, vmax=vmax)
    ax3.set_title("YZ (x=mid)")
    ax3.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    input_volume = r"/root/autodl-tmp/durian/001/output/recon_512.npy"  # 输入的3D图像路径
    output_directory = r"/root/autodl-tmp/durian/001/output"    # 输出文件夹路径
    prefix = '_recon_512'  # 输出文件的前缀

    p_low = 0.5
    p_high = 99.5
    dpi = 180

    os.makedirs(output_directory, exist_ok=True)
    vol = np.load(input_volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
    
    z, y, x = vol.shape
    z0, y0, x0 = z // 2, y // 2, x // 2

    # Center slices
    xy = vol[z0, :, :]
    xz = vol[:, y0, :]
    yz = vol[:, :, x0]

    vmin, vmax = compute_window(vol, p_low, p_high)

    # Save individual slices
    p_xy = os.path.join(output_directory, f"{prefix}_slice_xy_z{z0}.png")
    p_xz = os.path.join(output_directory, f"{prefix}_slice_xz_y{y0}.png")
    p_yz = os.path.join(output_directory, f"{prefix}_slice_yz_x{x0}.png")
    save_slice(xy, p_xy, f"XY (z={z0})", vmin, vmax, dpi)
    save_slice(xz, p_xz, f"XZ (y={y0})", vmin, vmax, dpi)
    save_slice(yz, p_yz, f"YZ (x={x0})", vmin, vmax, dpi)

    # Save montage
    p_m = os.path.join(output_directory, f"{prefix}_montage.png")
    save_montage(xy, xz, yz, p_m, vmin, vmax, dpi, title=f"{prefix}: window=[{vmin:.4g}, {vmax:.4g}]  shape={vol.shape}")

    # Print summary
    print("Done.")
    print("Volume:", input_volume)
    print("Shape:", vol.shape, "dtype:", vol.dtype)
    print("Window:", vmin, vmax)
    print("Saved:")
    print(" ", p_xy)
    print(" ", p_xz)
    print(" ", p_yz)
    print(" ", p_m)
    
if __name__ == "__main__":
    main()
