# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import imageio

# ========= 基本配置 =========
folder_path = './haptic_field'   # 你的切片数据目录（每个txt: x, y, val）
output_filename_scatter = 'verification_scatter_plot_python.png'
output_filename_image   = 'direct_map_image_from_val_1280x800.png'

# 目标图像尺寸（像素）
image_width_pixels  = 1280
image_height_pixels = 800

# 像素邻域半径：2 会更连贯（可调 1~3）
point_radius = 2

# 自定义调色板（与你之前一致）
colorsmap2 = ["#0B3E73", "#66A4C7", "#F1F5F6", "#E28B6F", "#7C1826"]
custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", colorsmap2)
# 预采样成 256 色条，便于快速索引
cmap_lut = (np.array([custom_cmap(i/255.0)[:3] for i in range(256)])).astype(float)  # (256,3) in [0,1]

# 背景颜色（选一个较柔和的中间色，避免太亮/太暗）
background_rgb = cmap_lut[127, :]   # 也可换成 very light: np.array([0.98,0.98,0.98])

# ========= 1) 查找与排序 =========
print(f"Looking for *.txt files in: {Path(folder_path).resolve()}")
p = Path(folder_path)
file_list_initial = sorted(list(p.glob('*.txt')))
if not file_list_initial:
    raise FileNotFoundError("No *.txt files found.")

# 用文件名中的“最后一个数字”排序（如 xxx_123.txt）
file_numbers, valid_files = [], []
for f_path in file_list_initial:
    match = re.findall(r'\d+', f_path.name)
    if match:
        try:
            file_numbers.append(int(match[-1]))
            valid_files.append(f_path)
        except ValueError:
            continue
if valid_files:
    sorted_idx = np.argsort(file_numbers)
    sorted_files = [valid_files[i] for i in sorted_idx]
    print(f"Found and sorted {len(sorted_files)} files.")
else:
    print("Warning: No numbered files. Using alpha sort.")
    sorted_files = file_list_initial

# ========= 2) 读取与拼接 =========
xs, ys, vs = [], [], []
print("\nImporting data (expecting 3 columns per file: x, y, val)...")
for fp in sorted_files:
    try:
        data = np.loadtxt(fp, delimiter=',')
        if data.ndim == 1:  # 单行情况
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            print(f"  Skip (need 3 cols): {fp.name}")
            continue
        xs.append(data[:, 0])
        ys.append(data[:, 1])
        vs.append(data[:, 2])
    except Exception as e:
        print(f"  Skip (read error): {fp.name}, {e}")

if not xs:
    raise RuntimeError("No valid 3-column data loaded.")

# 展平成一维点集（列优先：按切片顺序拼接）
x_vec = np.concatenate(xs)    # (N_total,)
y_vec = np.concatenate(ys)    # (N_total,)
val   = np.concatenate(vs)    # (N_total,)

N_total = x_vec.shape[0]
print(f"\nLoaded points: {N_total}")

# ========= 3) 自适应坐标范围（关键：避免“只占中间一小块”） =========
x_min, x_max = float(np.min(x_vec)), float(np.max(x_vec))
y_min, y_max = float(np.min(y_vec)), float(np.max(y_vec))
pad_x = 0.05 * (x_max - x_min + 1e-12)
pad_y = 0.05 * (y_max - y_min + 1e-12)
x_limits = np.array([x_min - pad_x, x_max + pad_x])
y_limits = np.array([y_min - pad_y, y_max + pad_y])
print(f"Adaptive x/y limits: x={x_limits}, y={y_limits}")

# ========= 4) 颜色映射（第三列 val 直接映射到色图） =========
# 假设 val 已归一化到 [0,1]；若不确定，可放开下一行做 min-max 归一化
# val = (val - val.min()) / (val.max() - val.min() + 1e-12)
idx = np.floor(np.clip(val, 0.0, 1.0) * 255.999).astype(int)
point_colors = cmap_lut[idx, :]  # (N_total, 3) in [0,1]

# ========= 5) 可选散点验证图 =========
print(f"\nCreating verification scatter plot (saving to {output_filename_scatter})...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_vec, y_vec, c=point_colors, s=5, marker='o', edgecolors='none')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('Verification Scatter Plot (color=val mapped)')
    ax.axis('equal'); ax.set_xlim(x_limits); ax.set_ylim(y_limits)
    ax.grid(True, linestyle=':', alpha=0.6); ax.set_facecolor((0.98, 0.98, 0.98))
    fig.savefig(output_filename_scatter, dpi=400, bbox_inches='tight')
    print("Scatter plot saved.")
except Exception as e:
    print(f"  Warning: scatter plot failed. Error: {e}")
finally:
    try:
        plt.close(fig)
    except:
        pass

# ========= 6) 直接像素映射成图 =========
print(f"\nCreating image via direct pixel mapping: {image_width_pixels}x{image_height_pixels} ...")

# 初始化背景（柔和中间色，避免刺眼/过亮）
final_image = np.ones((image_height_pixels, image_width_pixels, 3), dtype=float) * background_rgb.reshape(1,1,3)

# 映射到像素坐标（线性缩放 + round + 边界裁剪）
px = np.round((x_vec - x_limits[0]) / (x_limits[1] - x_limits[0]) * (image_width_pixels  - 1)).astype(int)
py = np.round((y_vec - y_limits[0]) / (y_limits[1] - y_limits[0]) * (image_height_pixels - 1)).astype(int)
px = np.clip(px, 0, image_width_pixels  - 1)
py = np.clip(py, 0, image_height_pixels - 1)

# 逐点画小邻域（覆盖写入，最小改动、不做加权累积）
print(f"Painting pixel neighborhoods with radius={point_radius} ...")
for i in range(N_total):
    cy, cx = py[i], px[i]
    color = point_colors[i]
    r0, r1 = cy - point_radius, cy + point_radius + 1
    c0, c1 = cx - point_radius, cx + point_radius + 1
    r0c, r1c = max(0, r0), min(image_height_pixels, r1)
    c0c, c1c = max(0, c0), min(image_width_pixels,  c1)
    if r0c < r1c and c0c < c1c:
        final_image[r0c:r1c, c0c:c1c, :] = color

# y 轴向上显示（图像坐标与数学坐标差异）
final_image = np.flipud(final_image)

# ========= 7) 保存 =========
print(f"Saving image to {output_filename_image} ...")
final_u8 = (np.clip(final_image, 0.0, 1.0) * 255).astype(np.uint8)
imageio.imwrite(output_filename_image, final_u8)
print("Done.")
