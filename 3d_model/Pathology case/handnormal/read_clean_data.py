import numpy as np
import matplotlib.pyplot as plt
import os
import re
import cv2



source_dir = './interpolation_pixel_friction/left_XYZ'
target_dir = './interpolation_pixel_friction/txt_clean_test1_XYZ'

os.makedirs(target_dir,exist_ok=True)


txt_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
sorted_txt_files = sorted(txt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))


for filename in sorted_txt_files:
    file_path = os.path.join(source_dir, filename)
    data = np.loadtxt(file_path, delimiter=',')

    x = data[:, 0]
    y_ori = data[:, 1]
    y = data[:, 2]
    max_val = np.max(y)
    max_indices = np.where(y == max_val)[0]
    # Normalize y for threshold detection
    y_nor = (y - np.min(y)) / (np.max(y) - np.min(y))

    y_round = np.round(y, 4)
    x_round = np.round(x, 4)
    y_ori_round = np.round(y_ori,4)

    match = re.search(r'\d+', filename)
    if match:
        number = int(match.group())
    else:
        number = -1


    # Compute 98th percentile
    # Compute thresholds based on percentiles
    lower_bound = np.percentile(y_round , 0.2)
    upper_bound = np.percentile(y_round , 99.8)
    keep_mask = (y_round >= lower_bound) & (y_round <= upper_bound)
    x_clean = x_round[keep_mask]
    y_clean = y_round[keep_mask]
    y_ori_clean = y_ori_round[keep_mask]


    max_val_clean = np.max(y_clean)
    max_indices_clean = np.where(y_clean == max_val_clean)[0]
    # Normalize cleaned y
    y_clean_nor = (y_clean - np.min(y_clean)) / (np.max(y_clean) - np.min(y_clean))

    # Save cleaned data
    cleaned_data = np.column_stack((x_clean,y_ori_clean, y_clean_nor))
    if 'gradient_re' in filename:
        new_filename = filename.replace('gradient_re', 'pixel_friction')
    else:
        new_filename = filename  #
    save_path = os.path.join(target_dir, new_filename)
    np.savetxt(save_path, cleaned_data, delimiter=',')
    print(f"Saved cleaned file: {save_path}")

    plt.figure()
    # plt.plot(x_filled[0:-1], slope_ori_pixel_re,'b-o')
    plt.plot(x_clean, y_clean_nor, 'b-o')
    # plt.plot(x_filled[0:-1], slope_list_array,'b-o')
    # plt.plot(x_sorted,y_sorted)
    # plt.plot(x_filled, y_filled_nor, 'r-o')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(6, 4) 
    save_fig_path = os.path.join(target_dir, filename[:-4] + '.png') # 设置尺寸为6x4英寸
    plt.savefig(save_fig_path, dpi=300)