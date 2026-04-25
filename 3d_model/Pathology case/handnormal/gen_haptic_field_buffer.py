import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors # For to_rgb and Normalize
import matplotlib.cm     # For ScalarMappable
from scipy.interpolate import interp1d
from pathlib import Path
import imageio # Use imageio for robust saving
import time    # To time the Z-buffer loop
import argparse

parser = argparse.ArgumentParser(description="Process slope data for left/right finger slips")
parser.add_argument('--direction', default='r', choices=['l', 'r'], help='finger slips direction: l=left, r=right')
args = parser.parse_args()
# === 映射方向到字符串子路径 ===
direction_map = {'l': 'left', 'r': 'right'}
direction_str = direction_map[args.direction]

# --- Configuration ---
folder_path = f'./haptic_field'  # <--- Set your folder path here

# --- Output Filenames ---
output_filename_scatter = f'verification_scatter_plot_v12_rev_order.png'
# <<< CHANGE 1: Filename reflects unified Z=0.3 color and Z-buffer >>>
output_filename_direct_zbuffer = f'direct_map_unified_z_bg_zbuffer_v12.png'

# Z value to target for override and background color determination
z_target_value = 0.3
z_mask_tolerance = 1e-6 # Tolerance for float comparison

# Point size (radius in pixels) for the direct mapping
point_radius = 1 # Use 1 for 3x3, adjust as needed

# --- Explicitly set target image dimensions ---
image_width_pixels = 1280
image_height_pixels = 800

# --- Data Coordinate Limits ---
x_limits_fixed = np.array([-1.6, 1.6])
y_limits_fixed = np.array([-1.0, 1.0])

# --- Custom Colormap Definition ---
hex_colors = ["#0B3E73", "#66A4C7", "#F1F5F6", "#E28B6F", "#7C1826"]

# --- Step 1: Find and Sort Files ---
print(f"Looking for *.txt files in: {Path(folder_path).resolve()}")
try:
    p = Path(folder_path)
    file_list_initial = sorted(list(p.glob('*.txt')))
    file_numbers = []
    valid_files = []
    for f_path in file_list_initial:
        match = re.findall(r'\d+', f_path.name)
        if match:
            try:
                file_numbers.append(int(match[-1]))
                valid_files.append(f_path)
            except ValueError:
                # Skip files where number extraction fails
                continue
    if valid_files:
        sorted_indices = np.argsort(file_numbers)
        sorted_files = [valid_files[i] for i in sorted_indices]
        print(f"Found and sorted {len(sorted_files)} files based on numbers.")
    elif file_list_initial:
        print(f"Warning: No files with extractable numbers. Using simple alphabetical sort.")
        sorted_files = file_list_initial
    else:
        raise FileNotFoundError(f"No *.txt files found in '{folder_path}'")
except Exception as e:
    print(f"Error finding or sorting files: {e}")
    exit()

# --- Step 2: Read and Concatenate Data ---
all_x_flat = []
all_y_flat = []
all_z_flat = []
slice_indices = []
print("\nImporting data (allowing variable points per slice)...")
for i, file_path in enumerate(sorted_files):
    try:
        data = np.loadtxt(file_path, delimiter=',', comments='#')
        # Handle empty files or files with insufficient data
        if data.ndim == 0 or data.size == 0:
            print(f"  Skipping empty file: {file_path.name}")
            continue
        if data.ndim == 1:
             # Reshape 1D array only if it has enough elements
             if data.size >= 3:
                 data = data.reshape(1, -1)
             else:
                 print(f"  Skipping file with insufficient columns (1D): {file_path.name}")
                 continue
        # Check for sufficient columns after potential reshape
        if data.shape[1] < 3:
            print(f"  Skipping file with insufficient columns: {file_path.name}")
            continue

        n_pts_slice = data.shape[0]
        all_x_flat.extend(data[:, 0])
        all_y_flat.extend(data[:, 1])
        all_z_flat.extend(data[:, 2])
        slice_indices.extend([i] * n_pts_slice) # Use file index 'i' as slice ID

    except Exception as e:
        print(f"  Warning: Skipping file {file_path.name} due to error: {e}")
        continue

if not all_x_flat:
    print("\nError: No valid data loaded after processing all files.")
    exit()

x_vec = np.array(all_x_flat)
y_vec = np.array(all_y_flat)
z_vec_original = np.array(all_z_flat) # Keep original Z
slice_indices = np.array(slice_indices)
total_points = len(x_vec)
n_slices_loaded = len(np.unique(slice_indices))

if total_points == 0:
     print("\nError: No data points available after loading.")
     exit()
print(f"\nData loaded and flattened: {total_points} points from {n_slices_loaded} slices.")


# --- Step 3: Prepare Colormap and Colors (Unified Background/Override) ---
# Build the interpolated colormap
n_colors = len(hex_colors)
try:
    custom_colormap_base = np.array([matplotlib.colors.to_rgb(h) for h in hex_colors])
except Exception as e:
    print(f"Error converting hex colors: {e}")
    exit()

cmap_indices_base = np.linspace(0, 1, n_colors)
cmap_indices_interp = np.linspace(0, 1, 256)
interp_func_r = interp1d(cmap_indices_base, custom_colormap_base[:, 0])
interp_func_g = interp1d(cmap_indices_base, custom_colormap_base[:, 1])
interp_func_b = interp1d(cmap_indices_base, custom_colormap_base[:, 2])
colormap_interp = np.vstack([
    interp_func_r(cmap_indices_interp),
    interp_func_g(cmap_indices_interp),
    interp_func_b(cmap_indices_interp)
]).T
print("Colormap prepared.")

# Calculate the SINGLE color for Z=0.3 based on GLOBAL Z range
# This color will be used for BOTH background and point override
print(f"\nDetermining unified color corresponding to absolute Z={z_target_value}...")
global_z_min = np.min(z_vec_original)
global_z_max = np.max(z_vec_original)
print(f"  Global Z range: [{global_z_min:.4f}, {global_z_max:.4f}]")

# <<< CHANGE 2: Store the calculated color for unified use >>>
unified_z03_color = None
if global_z_max > global_z_min:
    norm_target_z = (z_target_value - global_z_min) / (global_z_max - global_z_min)
    norm_target_z = np.clip(norm_target_z, 0.0, 1.0)
    target_z_color_idx = np.floor(norm_target_z * 255.999).astype(int)
    unified_z03_color = colormap_interp[target_z_color_idx, :]
    print(f"  Normalized Z={z_target_value} is {norm_target_z:.4f}, maps to index {target_z_color_idx}")
    print(f"  Unified Background/Z-Override Color: {unified_z03_color}")
else:
    # Handle case where all Z values are the same
    print(f"  Warning: Global Z range is zero. Using mid-colormap color as fallback.")
    unified_z03_color = colormap_interp[127, :] # Default to middle color

# Calculate Per-Slice Colors (used when Z is NOT close to target)
colors_rgb_per_slice = np.zeros((total_points, 3))
print("\nCalculating base colors based on per-slice Z-normalization...")
unique_slice_ids = np.unique(slice_indices)
for slice_id in unique_slice_ids:
    mask = (slice_indices == slice_id)
    # Skip if a slice ID has no points (shouldn't happen with current loading)
    if not np.any(mask):
         continue
    z_col = z_vec_original[mask]
    z_min, z_max = np.min(z_col), np.max(z_col)
    if z_max > z_min:
        z_norm = (z_col - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros(z_col.shape) # Assign 0 if range is zero

    z_idx = np.floor(z_norm * 255.999).astype(int)
    z_idx = np.clip(z_idx, 0, 255)
    colors_rgb_per_slice[mask, :] = colormap_interp[z_idx, :]
print("Per-slice base color calculation complete.")

# Determine final point colors: Use UNIFIED color if Z ≈ target, otherwise use PER-SLICE color
# <<< CHANGE 3: Use unified_z03_color in np.where >>>
print(f"Applying UNIFIED background color for points where Z ≈ {z_target_value}...")
# NOTE: This will make these points blend into the background!
is_target_z = np.isclose(z_vec_original, z_target_value, atol=z_mask_tolerance)
num_z_override = np.sum(is_target_z)
print(f"  Applying background color to {num_z_override} points.")

point_colors = np.where(
    is_target_z[:, np.newaxis],                          # Condition
    np.array(unified_z03_color).reshape(1, 3), # True: Use UNIFIED color
    colors_rgb_per_slice                                 # False: Use Per-Slice color
)
print("Final point colors determined (Z=0.3 points will match background).")


# --- Step 4: Create Verification Scatter Plot ---
# Uses reversed plotting order and FINAL colors
print(f"\nCreating verification scatter plot ({output_filename_scatter})...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    marker_size = 1
    # Plot using final point colors (including the background color for Z=0.3 points)
    ax.scatter(
        x_vec[::-1],
        y_vec[::-1],
        s=marker_size,
        c=point_colors[::-1], # Use final calculated colors
        marker='o',
        edgecolors='none',
        rasterized=True
    )
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Scatter Plot (Final Colors, Reversed Plot Order)')
    ax.axis('equal')
    ax.set_xlim(x_limits_fixed)
    ax.set_ylim(y_limits_fixed)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_facecolor((0.98, 0.98, 0.98)) # Plot background
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = matplotlib.cm.ScalarMappable(
        cmap=matplotlib.colors.ListedColormap(colormap_interp),
        norm=norm
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Norm. Z (Per-Slice Basis)', shrink=0.75)
    fig.savefig(output_filename_scatter, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved.")
except Exception as e:
    print(f"Warning: Scatter plot failed: {e}")
finally:
    # Ensure plot is closed to free memory
    if 'fig' in locals():
        plt.close(fig)


# --- Step 5: Create Image using Z-Buffering ---
print(f'\nCreating direct map image ({output_filename_direct_zbuffer}) using Z-buffer...')

# <<< CHANGE 4: Initialize image with the UNIFIED color >>>
final_image_zbuf = np.ones(
    (image_height_pixels, image_width_pixels, 3), dtype=float) * np.array(unified_z03_color).reshape(1, 1, 3)

# Initialize Z-buffer with infinity
z_buffer = np.full(
    (image_height_pixels, image_width_pixels), np.inf, dtype=float)
print(f"Initialized image with unified background color and Z-buffer (shape {z_buffer.shape}).")

# Calculate target pixel coordinates (centers)
print("Mapping data points to pixel coordinates...")
px = np.round(
    (x_vec - x_limits_fixed[0]) / (x_limits_fixed[1] - x_limits_fixed[0]) * (image_width_pixels - 1)
).astype(int)
py = np.round(
    (y_vec - y_limits_fixed[0]) / (y_limits_fixed[1] - y_limits_fixed[0]) * (image_height_pixels - 1)
).astype(int)
# Clip coordinates to be within image bounds
px = np.clip(px, 0, image_width_pixels - 1)
py = np.clip(py, 0, image_height_pixels - 1)
print("Pixel coordinates calculated.")

# Loop through points (forward order is fine for Z-buffer)
print(f"Processing points with Z-buffer (radius={point_radius}, {total_points} points)...")
start_time = time.time()
points_processed = 0
#for i in range(total_points):
for i in range(total_points - 1, -1, -1): 
    z_point = z_vec_original[i] # Use original Z for depth check
    # Use final color (which might be background color if Z approx 0.3)
    color_point = point_colors[i]
    center_py, center_px = py[i], px[i]

    # Calculate neighborhood boundaries (clipped)
    row_start = max(0, center_py - point_radius)
    row_end = min(image_height_pixels, center_py + point_radius + 1)
    col_start = max(0, center_px - point_radius)
    col_end = min(image_width_pixels, center_px + point_radius + 1)

    # Iterate through pixels IN the neighborhood
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            # Z-test: Is this point closer than what's already at (r, c)?
            if np.isinf(z_buffer[r, c]):
                # 首次赋值：直接填
                final_image_zbuf[r, c] = color_point
                z_buffer[r, c] = z_point
            else:
                if (args.direction == 'r' and z_point > z_buffer[r, c]) or \
                   (args.direction == 'l' and z_point < z_buffer[r, c]):
                    final_image_zbuf[r, c] = color_point
                    z_buffer[r, c] = z_point

    points_processed += 1
    # Print progress periodically
    if points_processed % (total_points // 20 or 1) == 0:
         elapsed = time.time() - start_time
         print(f"  Processed {points_processed}/{total_points} points... ({elapsed:.1f}s)")

end_time = time.time()
print(f"Z-buffer processing complete. Time taken: {end_time - start_time:.2f} seconds.")


# --- Step 6: Final Flip and Save ---
print("\nApplying final vertical flip (np.flipud).")
final_image_flipped = np.flipud(final_image_zbuf)
print(f'Final image constructed ({final_image_flipped.shape}) and flipped.')

try:
    output_path = Path(output_filename_direct_zbuffer)
    print(f'\nSaving final image as {output_path.name}...')
    # Convert to uint8 for saving
    final_image_save = (np.clip(final_image_flipped, 0.0, 1.0) * 255).astype(np.uint8)
    # Sanity check shape before saving
    if final_image_save.shape != (image_height_pixels, image_width_pixels, 3):
        print(f"CRITICAL WARNING: Final image shape {final_image_save.shape} is incorrect!")
    imageio.imwrite(output_path, final_image_save)
    print(f'Image saved successfully.')
except ImportError:
    print("\nERROR: imageio library not found (`pip install imageio`). Cannot save image.")
except Exception as e:
    print(f"\nFATAL ERROR saving image: {e}")
    # Report shape if save failed after array creation
    if 'final_image_save' in locals():
         print(f"  Attempted to save array with shape: {final_image_save.shape}")

print("\nScript finished.")