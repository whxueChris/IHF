import win32pipe
import win32file
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import cv2
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
import time
import argparse
from scipy.signal import find_peaks


pipeName = r'\\.\pipe\TestPipe'
bufferSize = 4096
width =1280, height = 800 # Range of the TanvasTouch pixel
Xlim = 3.2 ,Ylim =2 # Range of the point cloud data


# Define posture options
posture_folders = {
    1: 'Pretraining', # pre-training
    2: 'Hands up',
    3: 'Horse stance',
    4: 'Stand',
    5: 'Yoga',
    6: 'Dancing',
    7: 'Gymnastics',
}

# Select the posture index 
selected_index = 7  # Change this number to select a different posture

# Validate index
if selected_index not in posture_folders:
    raise ValueError(f"Invalid index {selected_index}. Must be one of {list(posture_folders.keys())}")

# Set source and target folder paths
source_folder = f"./{posture_folders[selected_index]}/"
target_gray_folder = f"{source_folder}test_fingerpath/gray_re/"



def adaptive_amplify_slopes(slopes, amplification_factor, threshold_scale=1):
    """
    Amplifies the slopes of y with respect to x based on a dynamic threshold.
    Args:
    - x: x values of the data.
    - y: y values of the data.
    - amplification_factor: The factor by which to amplify the slopes.
    - threshold_scale: A scaling factor to apply to the threshold value.

    Returns:
    - amplified_slopes: The slopes after amplification.
    - threshold: The calculated threshold for amplification.
    """
    # Calculate slopes
    #slopes = np.gradient(y, x)

    # Determine a dynamic threshold based on the standard deviation of the slopes
    threshold = threshold_scale * np.std(slopes)

    # Identify regions where the absolute slope is below the threshold
    low_slope_indices = np.where(np.abs(slopes) < threshold)

    # Amplify the slopes in these regions
    amplified_slopes = slopes.copy()
    amplified_slopes[low_slope_indices] *= amplification_factor

    # Smooth the transitions for continuity
    smoothed_slopes = np.convolve(amplified_slopes, np.ones(3) / 3, mode='same')

    return smoothed_slopes, threshold


def fill_gaps_with_interpolation(x, y, gap_threshold, fixed_ratio):
    """
    Identifies gaps in x-values and fills them with interpolated x-values.
    Fills y-values with the smaller y-value of the two points at each end of the gap.

    Parameters:
    - x (np.array): The x-values of the data points.
    - y (np.array): The y-values of the data points.
    - gap_threshold (float): The threshold to identify what is considered a large gap.
    - fixed_ratio (float): The ratio used to determine the number of points for interpolation.

    Returns:
    - np.array: The x and y values after filling the gaps.
    """
    new_x = [x[0]]
    new_y = [y[0]]

    for i in range(1, len(x)):
        # Calculate the gap
        gap = x[i] - x[i - 1]
        if gap > gap_threshold:
            # Number of points to interpolate
            num_points = int(gap / fixed_ratio)

            # Interpolated x-values
            x_fill = np.linspace(x[i - 1], x[i], num=num_points + 2)[1:-1]

            # Choose the smaller y-value for filling
            y_fill_value = min(y[i - 1], y[i])
            y_fill = np.full_like(x_fill, y_fill_value)

            # Append the interpolated points
            new_x.extend(x_fill)
            new_y.extend(y_fill)

        # Append the current point
        new_x.append(x[i])
        new_y.append(y[i])

    return np.array(new_x), np.array(new_y)


while True:
    try:
        # 
        pipeHandle = win32file.CreateFile(pipeName,
                                          win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                                          0,
                                          None,
                                          win32file.OPEN_EXISTING,
                                          0,
                                          None)

        while True:
             # Create a handle to the named pipe
            result = win32pipe.PeekNamedPipe(pipeHandle, bufferSize)
            #print(result)
            if result != 0:
                # Data received from C#

                dataBytes = win32file.ReadFile(pipeHandle, bufferSize)[1]
                dataReceived = dataBytes.decode('utf-8')


                # Process the received data
                processed_data = dataReceived.upper()
                start_time = time.time()


                data = []

                parser = argparse.ArgumentParser()
                print("Last character:", dataReceived[-1])
                s = 0


                if int(dataReceived[-1]) == 0:
                    # Left to right
                    s = 0
                    parser.add_argument('--direction', default='l', help='finger slips direction')
                    args = parser.parse_args()
                else:
                    # right to left
                    s = 1
                    parser.add_argument('--direction', default='r', help='finger slips direction')
                    args = parser.parse_args()

                print("Initial value of s:",s)
                dataReceived = [float(num) for num in dataReceived.strip().split(',')]


                for i in range(0, len(dataReceived), 2):
                    x = dataReceived[i]
                    if i + 1 < len(dataReceived):
                        y = dataReceived[i + 1]
                        if x >= 0 and y >= 0:  
                            data.append([x,y])

                data_array = np.array(data)
                fixed_ratio = 3.2 / 1280
                # load the k-means the data
                txt_files = [file for file in os.listdir(source_folder) if file.endswith('.txt')]
                sorted_files = sorted(txt_files, key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+',
                                                                                                           x) else 0)  # sort the txt files with numbers

                kmeans_all = np.empty((0, 3))
                for i in range(len(sorted_files)):
                    file_path = os.path.join(source_folder, sorted_files[i])
                    filename = sorted_files[i][:-4]
                    #print('Processing the file is', filename)
                    data_temp = np.loadtxt(file_path, delimiter=',')
                    kmeans_all = np.vstack((kmeans_all, data_temp))

                finger_data = data_array
                data_x1 = finger_data[:, 0]
                data_y1 = finger_data[:, 1]
                data_y1 = height - data_y1
                result = []
                nor_datay1 = (data_y1 - 0) * (Ylim / (height)) - Ylim / 2
                nor_datax1 = (data_x1 - 0) * (Xlim / (width)) - Xlim / 2

                x_min = np.min(kmeans_all[:, 0])
                x_max = np.max(kmeans_all[:, 0])
                nor_finger1 = np.concatenate((nor_datax1.reshape(1, -1), nor_datay1.reshape(1, -1)), axis=0)
                nor_finger1 = np.transpose(nor_finger1)
                finger1_indices = (nor_finger1[:, 0] >= x_min) & (nor_finger1[:, 0] <= x_max)
                finger1_region = nor_finger1[finger1_indices]

                y_min1 = np.min(finger1_region[:, 1])
                y_max1 = np.max(finger1_region[:, 1])


                # Spatial alignment 
                kmeans_range = (kmeans_all[:, 1] >= y_min1) & (kmeans_all[:, 1] <= y_max1)
                kmeans_range_1 = kmeans_all[kmeans_range]
                kdtree = cKDTree(kmeans_range_1[:, :2])
                nearestIndices = kdtree.query(finger1_region[:, :2], k=1)[1]
                if len(kmeans_range_1) > 0:  # or kmeans.size > 0
                    nearestValues = kmeans_range_1[nearestIndices]
                    # Perform other operations with nearestValues and the array
                else:
                    print('Outside the forecast range!')
                    continue


                x_curve = nearestValues[:, 0]
                y_curve = nearestValues[:, 1]
                z_curve = nearestValues[:, 2]

                sort_indices = np.argsort(x_curve)
                x_sorted = x_curve[sort_indices]
                y_sorted = y_curve[sort_indices]
                z_sorted = z_curve[sort_indices]

                unique_indices = np.unique(x_sorted, return_index=True)[1]
                x_unique = x_sorted[np.sort(unique_indices)]
                y_unique = z_sorted[np.sort(unique_indices)]  # find the height of each curve
                x_curve1 = x_unique
                y_curve1 = y_unique

                k_clusters = 2  
                kmeans = KMeans(n_clusters=k_clusters, random_state=0)
                kmeans.fit(np.column_stack((x_curve1, y_curve1)))

                
                interp_results = []
                for cluster_label in range(k_clusters):
                    cluster_points = np.where(kmeans.labels_ == cluster_label)[0]
                    cluster_x = x_curve1[cluster_points]
                    cluster_y = y_curve1[cluster_points]

                    # caculate the diff of cluster_x, cauclate the region index of cluster x, find the maximum diff_x

                    cluster_x_diff = np.diff(cluster_x)
                    cluster_x_index = np.argsort(cluster_x_diff)[::-1]
                    max_x_diff = argrelextrema(cluster_x_diff, np.greater)

                    if cluster_x_index.size == 0:

                        cluster_x = np.append(cluster_x, 0)
                        cluster_y = np.append(cluster_y, 0)
                        cluster_x_index = np.array([0])  


                        cluster_x_diff = np.diff(cluster_x)
                        cluster_x_index = np.argsort(cluster_x_diff)[::-1]
                    if cluster_x_index.size > 1 :
                        dx_gap1 = cluster_x[cluster_x_index[0] + 1] - cluster_x[cluster_x_index[0]]
                        num_gap1 = round(dx_gap1 / fixed_ratio)
                        y_value_to_fill = min(cluster_y[cluster_x_index[0]], cluster_y[cluster_x_index[0] + 1])
                        y_filled_gap1 = np.repeat(y_value_to_fill, num_gap1, axis=0)
                    else:
                        dx_gap1 = 0 
                        num_gap1 = 0
                    #dx_gap1 = cluster_x[cluster_x_index[0] + 1] - cluster_x[cluster_x_index[0]]
                    
                    if cluster_x_index.size >= 2:
                        x_filled_gap1 = np.linspace(cluster_x[cluster_x_index[0]],
                                            cluster_x[cluster_x_index[0] + 1],
                                            num=num_gap1)
                    else:
                        x_filled_gap1 = np.array([])
                        y_filled_gap1 = np.array([])



                    # the value of x, adopt the number of points to interpolate it
                    if cluster_x_index[0] == 0:
                        dx_tmp = np.max(cluster_x[cluster_x_index[0] + 1:]) - np.min(cluster_x[cluster_x_index[0] + 1:])
                        n1 = round(dx_tmp / fixed_ratio)
                        # added by whxue 0303,0725
                        if n1<=0:
                            n1 = n1+1
                        x_interp1 = np.linspace(np.min(cluster_x[cluster_x_index[0] + 1:]),
                                                np.max(cluster_x[cluster_x_index[0] + 1:]), n1)

                        number_points_inter = len(cluster_x[cluster_x_index[0]+1:])
                        # check the number of the data points of interpolation
                        if len(cluster_x[cluster_x_index[0]+1:])>3:
                            #spline_init = UnivariateSpline(cluster_x, cluster_y, k=3)
                            spline_init = UnivariateSpline(cluster_x[cluster_x_index[0] + 1:], cluster_y[cluster_x_index[0] + 1:],
                                                        k=3)
                        elif len(cluster_x[cluster_x_index[0] + 1:]) > 2:
                            spline_init = UnivariateSpline(cluster_x[cluster_x_index[0] + 1:], cluster_y[cluster_x_index[0] + 1:],
                                                        k=2)
                        elif len(cluster_x[cluster_x_index[0] + 1:]) > 1:
                            spline_init = UnivariateSpline(cluster_x[cluster_x_index[0] + 1:], cluster_y[cluster_x_index[0] + 1:],
                                                        k=1)
                        spline_init.set_smoothing_factor(0.0003)
                        y_interp1 = spline_init(x_interp1)

                        x_interp = x_interp1
                        y_interp = y_interp1
                    else:
                        dx_tmp = np.max(cluster_x[:cluster_x_index[0] + 1]) - np.min(cluster_x[:cluster_x_index[0] + 1])
                        n1 = round(dx_tmp / fixed_ratio)
                   
                        if n1 <= 0:
                            n1 = n1 + 1
                        x_interp_left = np.linspace(np.min(cluster_x[:cluster_x_index[0]+1]), np.max(cluster_x[:cluster_x_index[0]+1]),n1)
                        number_points_left = len(cluster_x[:cluster_x_index[0] + 1])
                        if len(cluster_x[:cluster_x_index[0]+1])>3:

                            spline_left = UnivariateSpline(cluster_x[:cluster_x_index[0] + 1], cluster_y[:cluster_x_index[0] + 1],k=3)
                        elif len(cluster_x[:cluster_x_index[0] + 1]) > 2:
                            spline_left = UnivariateSpline(cluster_x[:cluster_x_index[0] + 1], cluster_y[:cluster_x_index[0] + 1],k=2)
                        elif len(cluster_x[:cluster_x_index[0] + 1]) > 1:
                            spline_left = UnivariateSpline(cluster_x[:cluster_x_index[0] + 1], cluster_y[:cluster_x_index[0] + 1], k=1)
                        #spline_left = UnivariateSpline(cluster_x[:cluster_x_index[0]], cluster_y[:cluster_x_index[0]], k=2)
                        spline_left.set_smoothing_factor(0.0003)
                        y_interp_left = spline_left(x_interp_left)  

                        dx_tmp2 = np.max(cluster_x[cluster_x_index[0] + 1:]) - np.min(cluster_x[cluster_x_index[0] + 1:])
                        n2 = round(dx_tmp2 / fixed_ratio)
                        if n2 <= 0:
                            n2 = n2 + 1
                        x_interp_right = np.linspace(np.min(cluster_x[cluster_x_index[0] + 1:]),
                                                    np.max(cluster_x[cluster_x_index[0] + 1:]), n2)

                        number_points_right =len(cluster_x[cluster_x_index[0] + 1:])

                        if len(cluster_x[cluster_x_index[0] + 1:]) > 3:

                            spline_right = UnivariateSpline(cluster_x[cluster_x_index[0] + 1:], cluster_y[cluster_x_index[0] + 1:], k=3)
                            spline_right.set_smoothing_factor(0.0003)
                            y_interp_right = spline_right(x_interp_right)
                        elif len(cluster_x[cluster_x_index[0] + 1:]) > 2:
                            spline_right = UnivariateSpline(cluster_x[cluster_x_index[0] + 1:], cluster_y[cluster_x_index[0] + 1:], k=2)
                            spline_right.set_smoothing_factor(0.0003)
                            y_interp_right = spline_right(x_interp_right)
                        elif len(cluster_x[cluster_x_index[0] + 1:]) > 1:
                            spline_right = UnivariateSpline(cluster_x[cluster_x_index[0] + 1:], cluster_y[cluster_x_index[0] + 1:],k=1)
                            spline_right.set_smoothing_factor(0.0003)
                            y_interp_right = spline_right(x_interp_right)
                        else:
                            #spline_right = UnivariateSpline(cluster_x, cluster_y, k=1)
                            y_right_edge = cluster_y[cluster_x_index[0] + 1:]
                            y_interp_right =np.full(x_interp_right.shape, y_right_edge)

                        x_interp = np.concatenate((x_interp_left, x_filled_gap1, x_interp_right))
                        y_interp = np.concatenate((y_interp_left, y_filled_gap1, y_interp_right))


                    interp_results.append(np.column_stack((x_interp, y_interp)))

                interp_data = np.vstack(interp_results)
                x = interp_data[:, 0]
                y = interp_data[:, 1]
                sort_indices = np.argsort(x)
                x_sorted = x[sort_indices]
                y_sorted = y[sort_indices]

                gap_threshold = 0.005  # example threshold for a large gap, needs to be defined based on the data
                fixed_ratio = 3.2 / 1280  
                # Assuming gap_threshold and fixed_ratio are defined,check the gap
                x_sorted, y_sorted = fill_gaps_with_interpolation(x_sorted, y_sorted, gap_threshold, fixed_ratio)



                if len(x_sorted) > 0:

                    xmin_index = np.argmin(x_sorted)

                    xmax_index = np.argmax(x_sorted)

                else:
                    continue

                xmin = x_sorted[xmin_index]
                xmax = x_sorted[xmax_index]

                dx_min = xmin - (-1.6)
                dx_max = 1.6 - xmax
                nx_min = round(dx_min / fixed_ratio)
                nx_max = round(dx_max / fixed_ratio)

                # Perform horizontal interpolation to fill values on the left side of xmin
                x_min_fill = np.linspace(-1.6, xmin, num=nx_min)
                y_min_fill = np.interp(x_min_fill, x_sorted, y_sorted)

                #  Perform horizontal interpolation to fill values on the right side of xmax
                x_max_fill = np.linspace(xmax, 1.6, num=nx_max)
                y_max_fill = np.interp(x_max_fill, x_sorted, y_sorted)


                # chose the maximum dx indices
                x_diff = np.diff(x_sorted)
                x_indices_tmp = np.argsort(x_diff)[::-1]
                x_indices = np.where(x_sorted > 0)
                dx_gap1 = x_sorted[x_indices_tmp[0] + 1] - x_sorted[x_indices_tmp[0]]  # the different gap

                num_gap1 = round(dx_gap1 / fixed_ratio)

                if len(x_indices_tmp) > 0 and len(x_indices) > 0 and len(x_sorted) > 0:
                    start_index = x_indices_tmp[0] + 1

                    if len(x_indices[0]) > 0:
                        end_index = x_indices[0][0]
                    else:
                        continue

                    if start_index < len(x_sorted) and end_index < len(x_sorted):
                        x_filled_gap1 = np.linspace(x_sorted[start_index], x_sorted[end_index], num=num_gap1)
                    else:
                        continue
                else:
                    continue
                # y_filled_gap = y_sorted[x_indices[0][0] - 1]
                # y_filled_gap = np.repeat(y_filled_gap, num_gap, axis=0)
                y_filled_gap1 = y_sorted[x_indices_tmp[0] + 1]
                y_filled_gap1 = np.repeat(y_filled_gap1, num_gap1, axis=0)

    
                dx_gap = x_sorted[x_indices[0][0]] - x_sorted[x_indices[0][0] - 1]
                num_gap = round(dx_gap / fixed_ratio)
                y_filled_gap = y_sorted[x_indices[0][0] - 1]
                y_filled_gap = np.repeat(y_filled_gap, num_gap, axis=0)

                #  Merge the filled data and refill the gap near x = 0
                x_filled = np.concatenate((x_min_fill, x_sorted[xmin_index:x_indices_tmp[0] + 1], x_filled_gap1,
                                           x_sorted[x_indices_tmp[0] + 1:xmax_index + 1], x_max_fill))
                y_filled = np.concatenate((y_min_fill, y_sorted[xmin_index:x_indices_tmp[0] + 1], y_filled_gap1,
                                           y_sorted[x_indices_tmp[0] + 1:xmax_index + 1], y_max_fill))


                filled_data = np.column_stack((x_filled, y_filled))
                y_filled_nor = (y_filled - min(y_filled)) / (max(y_filled) - min(y_filled))
                dx_xmin_fill = x_min_fill[-1] - x_min_fill[0]
                dx_xmax_fill = x_max_fill[-1] - x_max_fill[0]
                dx_xmin_pixel = int(dx_xmin_fill / fixed_ratio)
                dx_xmax_pixel = int(dx_xmax_fill / fixed_ratio)

                slope_list = []
                # calculate the digital stimuli
                for j in range(len(x_filled) - 1):
                    x1 = x_filled[j]
                    y1 = y_filled[j]
                    x2 = x_filled[j + 1]
                    y2 = y_filled[j + 1]


                    if x2 == x1:
                        slope = 0
                    else:
                        slope = (y2 - y1) / (x2 - x1)

                    slope_list.append(slope)


                print(s)
                if s == 1:
                   slope_list = [slopes * -1 for slopes in slope_list]
                   print('use s')


                slope_list_array = np.array(slope_list)
                slope_ori_min = min(slope_list)
                slope_ori_dis = max(slope_list) - slope_ori_min
                slope_ori_pixel = [int(((slopes - slope_ori_min) / slope_ori_dis) * 255) for slopes in slope_list]
                slope_ori_pixel_re = np.array(slope_ori_pixel)
                a1_slope = np.where(slope_ori_pixel_re == slope_ori_pixel_re[0])
                a1_slope_diff = np.diff(a1_slope)
                a1_slope_indices_tmp = np.where(a1_slope_diff > 1.0)
                slope_grade = np.gradient(slope_ori_pixel_re)
                aa = np.where(x_filled > x_max_fill[0])
                for i in range(len(slope_ori_pixel_re) - 1):
                    if i <= len(x_min_fill) - 1:
                        slope_ori_pixel_re[i] = 0
                    if i >= aa[0][0]:
                        slope_ori_pixel_re[i] = 0

                # Based on the constitutive mapping of TanvasTouch,
                # the digital stimuli must be transformed into grayscale images with intensity values ranging from 0 to 255.
                img4 = np.zeros((height, len(slope_ori_pixel_re), 3), dtype=np.uint8)
                gray_img = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
                result = np.zeros_like(gray_img)
                y_filled = np.concatenate((y_min_fill, y_sorted[xmin_index:x_indices[0][0] - 1], y_filled_gap,
                                           y_sorted[x_indices[0][0]:xmax_index + 1], y_max_fill))

                peaks_y, _ = find_peaks(y_filled, height=0)
                need_amplification = len(peaks_y) > 1

                if need_amplification:

                    print('The slope of this file need to be amplificated', filename)

                    amplified_slopes_new, dynamic_threshold = adaptive_amplify_slopes(slope_list_array,
                                                                                      amplification_factor=10)

                    amp_slope_ori_min = np.min(amplified_slopes_new)
                    amp_slope_ori_dis = np.max(amplified_slopes_new) - amp_slope_ori_min

                    amp_slope_ori_pixel = 255 * ((amplified_slopes_new - amp_slope_ori_min) / amp_slope_ori_dis)

                    amp_slope_ori_pixel = amp_slope_ori_pixel.astype(int)
                    for i in range(len(amp_slope_ori_pixel) - 1):
                        if args.direction == 'r':
                            if i >= len(amp_slope_ori_pixel) - len(x_max_fill):  # right_to_left
                                amp_slope_ori_pixel[i] = 0
                        else:
                            if i <= len(x_min_fill):
                                amp_slope_ori_pixel[i] = 0

                    for j in range(len(amp_slope_ori_pixel) - 1):
                        result[:, j] = gray_img[:, j] + int(amp_slope_ori_pixel[j])

                    if args.direction == 'r':
                        for j in range(len(slope_ori_pixel_re)):
                            result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])
                        save_png_path = os.path.join(target_gray_folder, 'fingerpath_inter_slope.png')
                    else:
                        for j in range(len(slope_ori_pixel_re)):
                            result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])
                        save_png_path = os.path.join(target_gray_folder, 'fingerpath_inter_slope.png')

                    cv2.imwrite(save_png_path, result)

                else:
                    for j in range(len(slope_ori_pixel_re)):
                        for j in range(len(slope_ori_pixel_re)):
                            result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])
                        result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])

                    if args.direction == 'r':
                        for j in range(len(slope_ori_pixel_re)):
                            result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])
                        save_png_path = os.path.join(target_gray_folder, 'fingerpath_inter_slope.png')
                    else:
                        for j in range(len(slope_ori_pixel_re)):
                            result[:, j] = gray_img[:, j] + int(slope_ori_pixel_re[j])
                        save_png_path = os.path.join(target_gray_folder, 'fingerpath_inter_slope.png')

                    cv2.imwrite(save_png_path, result)

                end_time = time.time()


                print(f'Execution time: %s milliseconds' % (( end_time - start_time)*1000))



            # Pause for a while before reading data again
            time.sleep(1)  # Adjust the wait time as needed

        # Close the pipe handle
        win32file.CloseHandle(pipeHandle)
    except FileNotFoundError:
        #  If the named pipe is not found, wait for a while before the next connection attempt
        time.sleep(1)  # Adjust the wait time as needed