import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import re
import argparse
# 设置路径
input_folder = './haptic_field'
output_folder = './haptic_field/reconstruct_clean'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument('--direction',default='l',help='finger slips direction')
args = parser.parse_args()

txt_files = [f for f in os.listdir(input_folder ) if f.endswith('.txt')]
sorted_txt_files = sorted(txt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
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
    baseline =0.3
    rgb_rows = []
    all_y_list = []
    for file_name in sorted_txt_files:
        file_path = os.path.join(input_folder, file_name)
        data = np.loadtxt(file_path, delimiter=',')
        y_norm = data[:, 1]# 假设已经归一化

        if args.direction == 'r':
             y_processed = np.where(y_norm == baseline, -1*(1-baseline), -1 * y_norm)
             all_y_list.append(y_processed)
        else:
             all_y_list.append(y_norm)  
        #all_y_list.append(y_norm)   
        '''
        matching_mask = (y_norm ==y_norm[0] )
        if not (0.4 < y_norm[0] < 0.5):
            y_norm_new = y_norm.copy()
            y_norm_new[matching_mask] = 0.4
        else:
            y_norm_new = y_norm.copy()
        # RGB 映射
        y_rgb = (custom_cmap(y_norm_new)[:, :3] * 255).astype(np.uint8)  # shape: (1280, 3)
        rgb_rows.append(y_rgb[np.newaxis, :, :])  # shape: (1, 1280, 3)
        ''' 
    all_y_matrix = np.array(all_y_list)
    min_val = np.min(all_y_matrix)
    max_val = np.max(all_y_matrix)
    all_y_global_norm = (all_y_matrix - min_val) / (max_val - min_val)

    #heatmap_img = np.vstack(rgb_rows)  # shape: (num_files, 1280, 3)
    heatmap_img = (custom_cmap(all_y_global_norm)[:, :, :3] * 255).astype(np.uint8) 
    heatmap_img = np.flipud(heatmap_img)  


    #Image.fromarray(heatmap_img).save(output_image)
    #fig, ax = plt.subplots(figsize=(8, 5))
    fig, ax = plt.subplots(figsize=(1280/300, 800/300),dpi=300)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(heatmap_img, aspect='auto')
    ax.axis('off')
    plt.savefig(output_image, dpi=300, pad_inches=0)
    plt.close()

    #print(f'Heatmap image saved to: {output_image}')
    print(f'Heatmap image saved to: {output_image}')

def checkpixel(input_folder,output_image,slice_ids,width=1280,height=800):
    
    os.makedirs(output_folder, exist_ok=True)
    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", colorsmap2)
    for idx in slice_ids:
        filename = f'cluster_points_{idx}_inter_interzero_pixel_friction.txt'
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


    

generate_heatmap_resample_data(
    input_folder=input_folder,
    output_image=os.path.join(output_folder, 'stacked_haptic_field.png')
)



'''
checkpixel(input_folder=output_folder,
           output_image=os.path.join(output_folder),
           slice_ids=[108, 109, 110,111,112,113,114])'
'''