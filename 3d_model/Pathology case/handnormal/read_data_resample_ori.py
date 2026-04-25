import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import re
from scipy.ndimage import label


import argparse

parser = argparse.ArgumentParser(description="Process slope data for left/right finger slips")
parser.add_argument('--direction', default='l', choices=['l', 'r'], help='finger slips direction: l=left, r=right')
args = parser.parse_args()


# 设置路径
input_folder = './interpolation_pixel_friction/txt_clean_test1_XYZ'
output_folder = './interpolation_pixel_friction/txt_clean_resample_test1_XYZ'


output_folder2 = './haptic_field'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

os.makedirs(output_folder2, exist_ok=True)

# 设置目标长度
target_len = 1280
colorsmap2 = [
    #"#FFFFFF",  # 白色
    "#0B3E73",  # 非常浅的红色接近白色
    "#66A4C7",  # 浅红色
    "#F1F5F6",
    "#E28B6F",  # 较浅的红色
    "#7C1826"   # 深红色
]

def generate_heatmap_resample_data(input_folder,output_image):

    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", colorsmap2)

    # 获取所有 txt 文件并按数字排序
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    sorted_txt_files = sorted(txt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    rgb_rows = []
    for file_name in sorted_txt_files:
        file_path = os.path.join(input_folder, file_name)
        data = np.loadtxt(file_path, delimiter=',')
        y_norm = data[:, 1]  # 假设已经归一化

        # RGB 映射
        y_rgb = (custom_cmap(y_norm)[:, :3] * 255).astype(np.uint8)  # shape: (1280, 3)
        rgb_rows.append(y_rgb[np.newaxis, :, :])  # shape: (1, 1280, 3)

    heatmap_img = np.vstack(rgb_rows)  # shape: (num_files, 1280, 3)
    heatmap_img = np.flipud(heatmap_img)  

    #Image.fromarray(heatmap_img).save(output_image)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(heatmap_img, aspect='auto')
    ax.axis('off')
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    #print(f'Heatmap image saved to: {output_image}')
    print(f'Heatmap image saved to: {output_image}')

def checkpixel(input_folder,output_image,slice_ids,width=1280,height=800):
    
    os.makedirs(output_folder, exist_ok=True)
    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", colorsmap2)
    for idx in slice_ids:
        filename = f'cluster_points_{idx}_inter_interzero_gradient_re.txt'
        input_path = os.path.join(input_folder, filename)

        if not os.path.exists(input_path):
            print(f'Skip missing file: {filename}')
            continue

        data = np.loadtxt(input_path, delimiter=',')  # shape: (1280, 2)
        y_norm = data[:, 1]  # normalized values
        max_val = np.max(y_norm)
        min_val = np.min(y_norm)
        max_idx = np.argmax(y_norm)
        min_idx = np.argmin(y_norm)
        
        #plt.plot(data[:,0],y_norm,'b-o')
        #plt.show()
        # Map to RGB (0~1 → 0~255)
        rgb_colors = (custom_cmap(y_norm)[:, :3]).astype(np.float32)

        # Fill full image: each column has same RGB across height
        heatmap_rgb = np.zeros((height, width, 3), dtype=np.float32)
        for j in range(width):
            heatmap_rgb[:, j, :] = rgb_colors[j]

        heatmap_rgb_uint8 = (heatmap_rgb * 255).astype(np.uint8)

        # Save image
        '''
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(heatmap_rgb_uint8, aspect='auto')
        ax.axis('off')

        output_image = os.path.join(output_folder, f'slice_{idx}.png')
        plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f'Saved: slice_{idx}.png')'
        '''

def find_stable_y0_regions_multi(y, slope_threshold=0.005, padding=1):

    y0 = y[0]
    matching_mask = (y == y0)

    # 计算梯度
    dy = np.gradient(y)

    # 标记变化点（斜率大于阈值的位置）
    change_mask = np.abs(dy) > slope_threshold

    # 扩展变化点为变化区间
    expanded_change = np.zeros_like(change_mask)
    for i in np.where(change_mask)[0]:
        start = max(i - padding, 0)
        end = min(i + padding + 1, len(y))
        expanded_change[start:end] = 1

    # 标记每个变化区块
    labeled, num_features = label(expanded_change)

    # 汇总所有变化区块索引（合并多个区域）
    change_zone = np.zeros_like(y, dtype=bool)
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        change_zone |= region_mask

    # 最终掩码：只保留 y==y[0] 且在波动区间内的点
    final_mask = matching_mask & change_zone
    return final_mask


txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
sorted_txt_files = sorted(
    txt_files,
    key=lambda name: int(re.findall(r'\d+', name)[0]) if re.findall(r'\d+', name) else float('inf')
)

# 处理每个文件
for filename in sorted_txt_files:
    input_path = os.path.join(input_folder, filename)

    if not os.path.exists(input_path):
        print(f'Skip missing: {filename}')
        continue

    data = np.loadtxt(input_path, delimiter=',')
    x, y = data[:, 0], data[:, 2]
    y_ori = data[:,1]
    max_val = np.max(y)
    max_indices = np.where(y == max_val)[0]

    '''
    if len(max_indices) >= 2:
        idx1, idx2 = max_indices[0], max_indices[1]
        delete_start = max(idx2 - 1, 0)
        delete_end = min(idx2 + (idx2 - idx1), len(y))
        mask = np.ones(len(y), dtype=bool)
        mask[delete_start:delete_end] = False
        x = x[mask]
        y = y[mask]'
    '''

    n = len(x)

    if n > target_len:
        # 均匀下采样
        indices = np.linspace(0, n - 1, target_len).astype(int)
        x_new, y_new = x[indices], y[indices]
        y_ori_new = y_ori[indices]
    else:
        # 线性插值至目标点数
        interp_range = np.linspace(0, 1, n)
        interp_x = interp1d(interp_range, x, kind='linear')
        interp_y_ori = interp1d(interp_range, y_ori, kind='linear')
        interp_y = interp1d(interp_range, y, kind='linear')
        new_range = np.linspace(0, 1, target_len)
        x_new = interp_x(new_range)
        y_new = interp_y(new_range)
        y_ori_new = interp_y_ori(new_range)

    y_new2 = y_new.copy()
    mask_keep = find_stable_y0_regions_multi(y_new2)
    y_new2[~mask_keep & (y_new2 == y_new2[0])] = 0.3

    # 保存新数据
    output_path = os.path.join(output_folder, filename)
    output_path2 = os.path.join(output_folder2, filename)
    #np.savetxt(output_path, np.column_stack((x_new, y_new2)), delimiter=',',fmt='%.4f')
    np.savetxt(output_path2, np.column_stack((x_new, y_ori_new, y_new2)), delimiter=',',fmt='%.4f')

    print(f'Saved: {filename} ({n} → {target_len})')
    
    plt.figure()
    # plt.plot(x_filled[0:-1], slope_ori_pixel_re,'b-o')
    plt.plot(x_new, y_new, 'b-o')
    # plt.plot(x_filled[0:-1], slope_list_array,'b-o')
    # plt.plot(x_sorted,y_sorted)
    # plt.plot(x_filled, y_filled_nor, 'r-o')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(6, 4) 
    save_fig_path = os.path.join(output_folder, filename[:-4] + '_resample.png') # 设置尺寸为6x4英寸
    plt.savefig(save_fig_path, dpi=300)
    

'''
generate_heatmap_resample_data(
    input_folder=output_folder,
    output_image=os.path.join(output_folder, 'stacked_heatmap_body2.png')
)
'''


'''
checkpixel(input_folder=output_folder,
           output_image=os.path.join(output_folder),
           slice_ids=[108, 109, 110,111,112,113,114])
'''