import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors # For to_rgb and Normalize
import matplotlib.cm     # For ScalarMappable
# NOTE: scipy is only needed for interp1d in colormap generation
from scipy.interpolate import interp1d
from pathlib import Path
import imageio # Use imageio for robust saving
import argparse

parser = argparse.ArgumentParser(description="Process slope data for left/right finger slips")
parser.add_argument('--direction', default='l', choices=['l', 'r'], help='finger slips direction: l=left, r=right')
args = parser.parse_args()

# === 映射方向到字符串子路径 ===
direction_map = {'l': 'left', 'r': 'right'}
direction_str = direction_map[args.direction]

# --- Configuration ---
folder_path = f'./haptic_field_left_tmp'  # <--- Set your folder path here
output_filename_scatter = f'verification_scatter_plot_python.png' # Optional
# <<< CHANGE 1: Update output filename >>>
output_filename_direct = f'direct_map_image_unified_z_color_bg_1280x800_python.png'

# Z value to target for override and background color determination
z_target_value = 0.3
z_mask_tolerance = 1e-6 # Tolerance for float comparison when overriding points

# Point size (radius in pixels) for the direct mapping
point_radius = 1 # Radius 0=1x1, 1=3x3, 2=5x5, etc.

# --- Explicitly set target image dimensions ---
image_width_pixels = 1280
image_height_pixels = 800

# --- Data Coordinate Limits (Used for mapping points to pixels) ---
x_limits_fixed = np.array([-1.6, 1.6])
y_limits_fixed = np.array([-1.0, 1.0])

# --- Custom Colormap Definition ---
hex_colors = ["#0B3E73", "#66A4C7", "#F1F5F6", "#E28B6F", "#7C1826"]

# --- Step 1: Find and Sort Files ---
# (Code remains the same as before)
print(f"Looking for *.txt files in: {Path(folder_path).resolve()}")
try:
    p = Path(folder_path)
    file_list_initial = sorted(list(p.glob('*.txt')))
    file_numbers, valid_files = [], []
    for f_path in file_list_initial:
        match = re.findall(r'\d+', f_path.name)
        if match:
            try: file_numbers.append(int(match[-1])); valid_files.append(f_path)
            except ValueError: continue
    if valid_files:
        sorted_indices = np.argsort(file_numbers); sorted_files = [valid_files[i] for i in sorted_indices]
        print(f"Found and sorted {len(sorted_files)} files.")
    elif file_list_initial: print(f"Warning: No numbered files. Using alpha sort."); sorted_files = file_list_initial
    else: raise FileNotFoundError(f"No *.txt files found.")
except Exception as e: print(f"Error finding/sorting files: {e}"); exit()

# --- Step 2: Read and Concatenate Data ---
# (Code remains the same)
all_x_cols, all_y_cols, all_z_cols = [], [], []
n_points_ref = None
print("\nImporting data...")
for i, file_path in enumerate(sorted_files):
    try:
        data = np.loadtxt(file_path, delimiter=',')
        if data.ndim == 0: continue
        if data.ndim == 1: data = data.reshape(1, -1) if data.size >= 3 else None
        if data is None or data.shape[1] < 3: continue
        current_n_points = data.shape[0]
        if i == 0: n_points_ref = current_n_points
        elif current_n_points != n_points_ref: continue
        all_x_cols.append(data[:, 0]); all_y_cols.append(data[:, 1]); all_z_cols.append(data[:, 2])
    except Exception as e: continue
if not all_x_cols: print("\nError: No valid data loaded."); exit()
all_x = np.stack(all_x_cols, axis=1); all_y = np.stack(all_y_cols, axis=1); all_z = np.stack(all_z_cols, axis=1)
n_points, n_slices = all_z.shape
print(f"\nData concatenated: {n_points} points per slice, {n_slices} slices.")

# --- Step 3: Prepare Colormap and Determine Shared Z=0.3 Color ---

# Build the interpolated colormap (as before)
n_colors = len(hex_colors)
try: custom_colormap_base = np.array([matplotlib.colors.to_rgb(h) for h in hex_colors])
except:
    custom_colormap_base = np.zeros((n_colors, 3))
    try:
        for i, h in enumerate(hex_colors): custom_colormap_base[i,:] = tuple(int(h.lstrip('#')[j:j+2], 16) for j in (0,2,4))/255.0
    except: print("Hex conversion failed."); exit()
cmap_indices_base = np.linspace(0, 1, n_colors); cmap_indices_interp = np.linspace(0, 1, 256)
interp_func_r = interp1d(cmap_indices_base, custom_colormap_base[:, 0]); interp_func_g = interp1d(cmap_indices_base, custom_colormap_base[:, 1]); interp_func_b = interp1d(cmap_indices_base, custom_colormap_base[:, 2])
colormap_interp = np.vstack([interp_func_r(cmap_indices_interp), interp_func_g(cmap_indices_interp), interp_func_b(cmap_indices_interp)]).T
print("Colormap prepared.")

# <<< CHANGE 2: Calculate the SINGLE color for Z=0.3 based on GLOBAL Z range >>>
print(f"Determining color corresponding to absolute Z={z_target_value}...")
global_z_min = np.min(all_z)
global_z_max = np.max(all_z)
print(f"  Global Z range: [{global_z_min:.4f}, {global_z_max:.4f}]")

if global_z_max > global_z_min:
    # Normalize the target Z value within the global range
    norm_target_z = (z_target_value - global_z_min) / (global_z_max - global_z_min)
    norm_target_z = np.clip(norm_target_z, 0.0, 1.0) # Ensure it's within [0, 1]
    # Find the corresponding index in the 256-color map
    target_z_color_idx = np.floor(norm_target_z * 255.999).astype(int)
    shared_background_z_color = colormap_interp[target_z_color_idx, :]
    print(f"  Normalized Z={z_target_value} is {norm_target_z:.4f}, maps to index {target_z_color_idx}")
    print(f"  Shared Background/Z-Override Color: {shared_background_z_color}")
else:
    # Handle case where all Z values are the same
    print(f"  Warning: Global Z range is zero. Using mid-colormap color as fallback.")
    shared_background_z_color = colormap_interp[127, :] # Default to middle color

# --- Step 3b: Calculate Per-Slice Colors for Points NOT matching Z-target ---
x_vec = all_x.flatten('F'); y_vec = all_y.flatten('F'); z_vec_original = all_z.flatten('F')
total_points = n_points * n_slices
colors_rgb_per_slice = np.zeros((total_points, 3)) # Colors based on per-slice norm

print("\nCalculating colors based on per-slice Z-normalization (for non-override points)...")
for j in range(n_slices):
    z_col = all_z[:, j]; z_min, z_max = np.min(z_col), np.max(z_col)
    z_norm = (z_col - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(n_points)
    z_idx = np.floor(z_norm * 255.999).astype(int); z_idx = np.clip(z_idx, 0, 255)
    start_idx, end_idx = j * n_points, (j + 1) * n_points
    colors_rgb_per_slice[start_idx:end_idx, :] = colormap_interp[z_idx, :]
print("Per-slice color calculation complete.")

# --- (Optional) Step 4: Create Verification Scatter Plot ---
# (Uses per-slice colors for visualization consistency)
print(f"\nCreating verification scatter plot (saving to {output_filename_scatter})...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use per-slice colors here for the scatter plot itself
    ax.scatter(x_vec, y_vec, s=5, c=colors_rgb_per_slice, marker='o', edgecolors='none')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('Verification Scatter Plot (Per-Slice Colors)')
    ax.axis('equal'); ax.set_xlim(x_limits_fixed); ax.set_ylim(y_limits_fixed)
    ax.grid(True, linestyle=':', alpha=0.6); ax.set_facecolor((0.98, 0.98, 0.98))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1); sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(colormap_interp), norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Normalized Z (Slice)', shrink=0.75)
    fig.savefig(output_filename_scatter, dpi=500, bbox_inches='tight'); print(f"Scatter plot saved.")
except Exception as e: print(f"Warning: Scatter plot failed. Error: {e}")
finally: plt.close(fig) if 'fig' in locals() else None


# --- Step 5: Create Image by Direct Pixel Mapping ---
print(f'\nCreating {image_width_pixels} x {image_height_pixels} image via direct pixel mapping...')

# <<< CHANGE 3: Initialize with the SHARED calculated Z color >>>
final_image_direct = np.ones((image_height_pixels, image_width_pixels, 3), dtype=float) * np.array(shared_background_z_color).reshape(1, 1, 3)
print(f"Initialized image with shared background color: {shared_background_z_color}")

# Calculate target pixel coordinates (centers) for all data points
print("Mapping data points to pixel coordinates...")
px = np.round((x_vec - x_limits_fixed[0]) / (x_limits_fixed[1] - x_limits_fixed[0]) * (image_width_pixels - 1)).astype(int)
py = np.round((y_vec - y_limits_fixed[0]) / (y_limits_fixed[1] - y_limits_fixed[0]) * (image_height_pixels - 1)).astype(int)
px = np.clip(px, 0, image_width_pixels - 1); py = np.clip(py, 0, image_height_pixels - 1)
print("Pixel coordinates calculated.")

# Determine the color for each point: Use SHARED color if Z ≈ 0.3, otherwise use PER-SLICE color
print(f"Checking Z values for override (Target ≈ {z_target_value})...")
is_target_z = np.isclose(z_vec_original, z_target_value, atol=z_mask_tolerance)
num_z_override = np.sum(is_target_z)
if num_z_override > 0: print(f"  Found {num_z_override} points to override with shared Z color.")
else: print(f"  No points found with Z ≈ {z_target_value}.")

# <<< CHANGE 4: Use shared color for True, per-slice colors for False >>>
point_colors = np.where(is_target_z[:, np.newaxis], # Condition
                        np.array(shared_background_z_color).reshape(1, 3), # True: Shared Z color
                        colors_rgb_per_slice) # False: Per-Slice color

# Assign colors to pixel NEIGHBORHOODS
print(f"Assigning colors to pixel neighborhoods (radius={point_radius})...")
for i in range(total_points):
    center_py, center_px, color = py[i], px[i], point_colors[i]
    row_start, row_end = center_py - point_radius, center_py + point_radius + 2
    col_start, col_end = center_px - point_radius, center_px + point_radius + 2
    row_start_c, row_end_c = max(0, row_start), min(image_height_pixels, row_end)
    col_start_c, col_end_c = max(0, col_start), min(image_width_pixels, col_end)
    if row_start_c < row_end_c and col_start_c < col_end_c:
        final_image_direct[row_start_c:row_end_c, col_start_c:col_end_c, :] = color
print("Pixel neighborhood colors assigned.")


# --- Step 6: Final Flip and Save ---
print("\nApplying final vertical flip (np.flipud).")
final_image_flipped = np.flipud(final_image_direct)
print(f'Final image constructed ({final_image_flipped.shape}) and flipped.')

try:
    output_path = Path(output_filename_direct)
    print(f'\nAttempting to save the final {image_height_pixels}x{image_width_pixels} image as {output_path.name}...')
    final_image_save = (np.clip(final_image_flipped, 0.0, 1.0) * 255).astype(np.uint8)
    if final_image_save.shape != (image_height_pixels, image_width_pixels, 3):
         print(f"CRITICAL WARNING: Final image shape {final_image_save.shape} mismatch!")
    imageio.imwrite(output_path, final_image_save)
    print(f'Image saved successfully to: {output_path.resolve()}')
except ImportError: print("\nERROR: imageio not found (`pip install imageio`).")
except Exception as e:
    print(f"\nFATAL ERROR: Could not save the final image. Error: {e}")
    if 'final_image_save' in locals(): print(f"  Matrix shape was: {final_image_save.shape}")

print("\nScript finished.")