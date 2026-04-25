import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors # For to_rgb and Normalize
import matplotlib.cm     # For ScalarMappable
# NOTE: scipy.interpolate and scipy.ndimage are NOT needed for this approach
from pathlib import Path
import imageio # Use imageio for robust saving

# --- Configuration ---
folder_path = './haptic_field' # <--- Set your folder path here
output_filename_scatter = 'verification_scatter_plot_python.png' # Optional verification plot
output_filename_direct = 'direct_map_image_1280x800_python.png' # Final output

# Background color for pixels not hit by any data point
background_color = [1.0, 1.0, 1.0] # White background

# Color for points where original Z was approximately z_target_value
z_target_value = 0.3
z_mask_color = [1.0, 1.0, 1.0] # White
z_mask_tolerance = 1e-6 # Tolerance for float comparison

# --- Explicitly set target image dimensions ---
image_width_pixels = 1280
image_height_pixels = 800

# --- Data Coordinate Limits (Used for mapping points to pixels) ---
x_limits_fixed = np.array([-1.6, 1.6])
y_limits_fixed = np.array([-1.0, 1.0])

# --- Custom Colormap Definition ---
hex_colors = ["#0B3E73", "#66A4C7", "#F1F5F6", "#E28B6F", "#7C1826"]

# --- Step 1: Find and Sort Files ---
# (Code remains the same as before - finding and sorting *.txt files)
print(f"Looking for *.txt files in: {Path(folder_path).resolve()}")
try:
    p = Path(folder_path)
    file_list_initial = sorted(list(p.glob('*.txt')))
    file_numbers, valid_files = [], []
    for f_path in file_list_initial:
        match = re.findall(r'\d+', f_path.name)
        if match:
            try:
                file_numbers.append(int(match[-1]))
                valid_files.append(f_path)
            except ValueError: continue # Skip if number conversion fails
        # else: Skip files without numbers silently or add warning
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
    print(f"Error finding or sorting files: {e}"); exit()

# --- Step 2: Read and Concatenate Data ---
# (Code remains the same - reading files into all_x, all_y, all_z)
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
        all_x_cols.append(data[:, 0])
        all_y_cols.append(data[:, 1])
        all_z_cols.append(data[:, 2])
        # print(f"  File {file_path.name} imported ({data.shape[0]} points).") # Verbose
    except Exception as e: continue # Skip bad files
if not all_x_cols: print("\nError: No valid data loaded."); exit()
all_x = np.stack(all_x_cols, axis=1)
all_y = np.stack(all_y_cols, axis=1)
all_z = np.stack(all_z_cols, axis=1)
n_points, n_slices = all_z.shape
print(f"\nData concatenated: {n_points} points per slice, {n_slices} slices.")

# --- Step 3: Prepare Colors Based on Per-Slice Z-Normalization ---
# (Code remains the same - calculating colors_rgb based on colormap and per-slice Z)
n_colors = len(hex_colors)
try:
    custom_colormap_base = np.array([matplotlib.colors.to_rgb(h) for h in hex_colors])
except: # Manual fallback
    custom_colormap_base = np.zeros((n_colors, 3))
    try:
        for i, h in enumerate(hex_colors): custom_colormap_base[i,:] = tuple(int(h.lstrip('#')[j:j+2], 16) for j in (0,2,4))/255.0
    except: print("Hex conversion failed."); exit()
from scipy.interpolate import interp1d # Still need this for colormap
cmap_indices_base = np.linspace(0, 1, n_colors)
cmap_indices_interp = np.linspace(0, 1, 256)
interp_func_r = interp1d(cmap_indices_base, custom_colormap_base[:, 0])
interp_func_g = interp1d(cmap_indices_base, custom_colormap_base[:, 1])
interp_func_b = interp1d(cmap_indices_base, custom_colormap_base[:, 2])
colormap_interp = np.vstack([interp_func_r(cmap_indices_interp), interp_func_g(cmap_indices_interp), interp_func_b(cmap_indices_interp)]).T

x_vec = all_x.flatten('F')
y_vec = all_y.flatten('F')
z_vec_original = all_z.flatten('F') # Keep original Z values flattened
total_points = n_points * n_slices
colors_rgb = np.zeros((total_points, 3))

print("\nCalculating colors based on per-slice Z-normalization...")
for j in range(n_slices):
    z_col = all_z[:, j]
    z_min, z_max = np.min(z_col), np.max(z_col)
    z_norm = (z_col - z_min) / (z_max - z_min) if z_max > z_min else np.zeros(n_points)
    z_idx = np.floor(z_norm * 255.999).astype(int); z_idx = np.clip(z_idx, 0, 255)
    start_idx, end_idx = j * n_points, (j + 1) * n_points
    colors_rgb[start_idx:end_idx, :] = colormap_interp[z_idx, :]
print("Color calculation complete.")

# --- (Optional) Step 4: Create Verification Scatter Plot ---
# (Code remains the same)
print(f"\nCreating verification scatter plot (saving to {output_filename_scatter})...")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_vec, y_vec, s=5, c=colors_rgb, marker='o', edgecolors='none')
    ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
    ax.set_title('Verification Scatter Plot (Per-Slice Z-Norm)')
    ax.axis('equal'); ax.set_xlim(x_limits_fixed); ax.set_ylim(y_limits_fixed)
    ax.grid(True, linestyle=':', alpha=0.6); ax.set_facecolor((0.98, 0.98, 0.98))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(colormap_interp), norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Normalized Z-Value (within each slice)', shrink=0.75)
    fig.savefig(output_filename_scatter, dpi=150, bbox_inches='tight')
    print(f"Verification scatter plot saved successfully.")
except Exception as e: print(f"Warning: Scatter plot failed. Error: {e}")
finally: plt.close(fig) if 'fig' in locals() else None


# --- Step 5: Create Image by Direct Pixel Mapping ---
print(f'\nCreating {image_width_pixels} x {image_height_pixels} image via direct pixel mapping...')

# Initialize the final image with the background color
# Ensure background color is float [0,1] for calculations
final_image_direct = np.ones((image_height_pixels, image_width_pixels, 3), dtype=float) * np.array(background_color).reshape(1, 1, 3)
print(f"Initialized image with background color: {background_color}")

# Calculate target pixel coordinates for all data points at once
print("Mapping data points to pixel coordinates...")
px = np.round((x_vec - x_limits_fixed[0]) / (x_limits_fixed[1] - x_limits_fixed[0]) * (image_width_pixels - 1)).astype(int)
py = np.round((y_vec - y_limits_fixed[0]) / (y_limits_fixed[1] - y_limits_fixed[0]) * (image_height_pixels - 1)).astype(int)

# Clip coordinates to ensure they are within image bounds
px = np.clip(px, 0, image_width_pixels - 1)
py = np.clip(py, 0, image_height_pixels - 1)
print("Pixel coordinates calculated and clipped.")

# Determine the color for each point, considering the Z=0.3 override
print(f"Checking Z values for override (Target ≈ {z_target_value})...")
is_target_z = np.isclose(z_vec_original, z_target_value, atol=z_mask_tolerance)
num_z_override = np.sum(is_target_z)
if num_z_override > 0:
    print(f"  Found {num_z_override} points to override with color: {z_mask_color}")
else:
    print(f"  No points found with Z ≈ {z_target_value}.")

# Choose color: z_mask_color if Z ≈ 0.3, otherwise the calculated colors_rgb
point_colors = np.where(is_target_z[:, np.newaxis], # Condition needs broadcasting
                        np.array(z_mask_color).reshape(1, 3), # True value (broadcast)
                        colors_rgb) # False value

# Assign the determined colors to the corresponding pixels
# Looping is clearer here than complex vectorized indexing if points overlap
print(f"Assigning colors to pixels ({total_points} points)...")
# Use (py, px) for (row, column) indexing in NumPy
final_image_direct[py, px, :] = point_colors
# Note: If multiple points map to the same pixel, the last one in the 'point_colors'
# array corresponding to that pixel will determine the final color.

print("Pixel colors assigned.")


# --- Step 6: Final Flip and Save ---
print("\nApplying final vertical flip (np.flipud).")
# Flip the image vertically for standard image orientation
final_image_flipped = np.flipud(final_image_direct)

print(f'Final image constructed ({final_image_flipped.shape}) and flipped.')

try:
    output_path = Path(output_filename_direct)
    print(f'\nAttempting to save the final {image_height_pixels}x{image_width_pixels} direct-mapped image as {output_path.name}...')

    # Convert the final FLIPPED float image [0.0, 1.0] to uint8 [0, 255]
    # Ensure values are clamped just in case, though they should be [0,1]
    final_image_save = (np.clip(final_image_flipped, 0.0, 1.0) * 255).astype(np.uint8)

    if final_image_save.shape != (image_height_pixels, image_width_pixels, 3):
         print(f"CRITICAL WARNING: Final image shape {final_image_save.shape} mismatch!")

    imageio.imwrite(output_path, final_image_save)
    print(f'Image saved successfully to: {output_path.resolve()}')

except ImportError:
     print("\nERROR: imageio library not found (`pip install imageio`). Cannot save image.")
except Exception as e:
    print(f"\nFATAL ERROR: Could not save the final image. Error: {e}")
    if 'final_image_save' in locals(): print(f"  Matrix shape was: {final_image_save.shape}")

print("\nScript finished.")