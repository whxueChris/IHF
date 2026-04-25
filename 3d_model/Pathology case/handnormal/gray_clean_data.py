import numpy as np
import matplotlib.pyplot as plt
import os
import re
import cv2


def first_last_diff_idx_float(arr, atol=1e-9):
    v0 = arr[0]
    mask = ~np.isclose(arr, v0, rtol=0.0, atol=atol)
    if not mask.any():
        return None, None, np.array([], dtype=int)
    idxs = np.flatnonzero(mask)
    return idxs[0], idxs[-1]

source_dir = './interpolation_pixel_friction/txt_clean_test1_XYZ'

txt_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
sorted_txt_files = sorted(txt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

target_dir = './gray_re2'

os.makedirs(target_dir,exist_ok=True)
height = 800


for filename in sorted_txt_files:
    file_path = os.path.join(source_dir, filename)
    data = np.loadtxt(file_path, delimiter=',')

    x = data[:, 0]
    y_ori = data[:, 1]
    slope = data[:, 2]

    slope = np.round(slope, 4)
    x = np.round(x, 4)
    y_ori = np.round(y_ori,4)

    s_min, s_max = slope.min(), slope.max()
    eq = (slope == slope[0])
    b = eq.astype(np.int8)
    d = np.diff(b)

    starts = np.where(d == -1)[0]                     # True->False
    first_idx = (starts[0] + 1) if starts.size else 0
    ends = np.where(d == +1)[0]                       # False->True
    last_idx = ends[-1] if ends.size else (len(slope) - 1)


    
    if s_max == s_min:
        # 全常数时直接置 0（也可改为 128 等常数）
        slope_ori_pixel_re = np.zeros_like(slope, dtype=np.uint8)
    else:
        slope_ori_pixel_re = np.rint((slope - s_min) / (s_max - s_min) * 255).astype(np.uint8)

    
    for i in range(len(slope_ori_pixel_re) ):
         if i <= first_idx or i >= last_idx:
                slope_ori_pixel_re[i] = 0

    img4 = np.zeros((height, len(slope_ori_pixel_re), 3), dtype=np.uint8)
    gray_img = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    result = np.zeros_like(gray_img)

    for j in range(len(slope_ori_pixel_re)):
        result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])

    m = re.match(r'^cluster_points_(\d+)_inter_interzero_pixel_friction\.txt$', filename)
    i_str = m.group(1)
    out_name = f'cluster_points_{i_str}_inter_slope.png'
    out_path = os.path.join(target_dir, out_name)

    # 保存为 8-bit 灰度图
    cv2.imwrite(out_path, result)