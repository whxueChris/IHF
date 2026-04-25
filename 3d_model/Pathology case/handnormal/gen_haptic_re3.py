import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors # For to_rgb and Normalize
import matplotlib.cm     # For ScalarMappable
from scipy.interpolate import interp1d
from pathlib import Path
import imageio # Use imageio for robust saving

# --- Configuration ---
folder_path = './haptic_field' # <--- Set your folder path here

# --- Output Filenames ---
output_filename_scatter = 'verification_scatter_plot_v7_rev_order.png'
output_filename_direct_rev = 'direct_map_unified_z_bg_re.png' # Reversed order output

# Z value to target for override and background color determination
z_target_value = 0.3
z_mask_tolerance = 1e-6 # Tolerance for float comparison when overriding points

# Point size (radius in pixels) for the direct mapping
# Set to 0 for single pixel assignment (no thickening)
# Set > 0 for thicker points
point_radius = 2# Example: Use 1 for 3x3 squares

# --- Explicitly set target image dimensions ---
image_width_pixels = 1280
image_height_pixels = 800

# --- Data Coordinate Limits ---
x_limits_fixed = np.array([-1.6, 1.6])
y_limits_fixed = np.array([-1.0, 1.0])

# --- Custom Colormap Definition ---
hex_colors = ["#0B3E73", "#66A4C7", "#F1F5F6", "#E28B6F", "#7C1826"]

# --- Step 1: Find and Sort Files ---
# (Using robust sorting - code unchanged)
print(f"Looking for *.txt files in: {Path(folder_path).resolve()}")
try:
    p = Path(folder_path); file_list_initial = sorted(list(p.glob('*.txt')))
    file_numbers, valid_files = [], []
    for f_path in file_list_initial:
        match = re.findall(r'\d+', f_path.name)
        if match:
            try: file_numbers.append(int(match[-1])); valid_files.append(f_path)
            except ValueError: continue
    if valid_files:
        sorted_indices = np.argsort(file_numbers); sorted_files = [valid_files[i] for i in sorted_indices]
        print(f"Found and sorted {len(sorted_files)} files based on numbers.")
    elif file_list_initial: print(f"Warning: No numbered files. Using alpha sort."); sorted_files = file_list_initial
    else: raise FileNotFoundError(f"No *.txt files found.")
except Exception as e: print(f"Error finding/sorting files: {e}"); exit()

# --- Step 2: Read and Concatenate Data ---
# (Using flexible list-based reading - code unchanged)
all_x_flat, all_y_flat, all_z_flat, slice_indices = [], [], [], []
print("\nImporting data (allowing variable points per slice)...")
for i, file_path in enumerate(sorted_files):
    try:
        data = np.loadtxt(file_path, delimiter=',', comments='#');
        if data.ndim==0 or data.size==0: continue;
        if data.ndim==1: data=data.reshape(1,-1) if data.size>=3 else None;
        if data is None or data.shape[1]<3: continue;
        n_pts_slice = data.shape[0]; all_x_flat.extend(data[:,0]); all_y_flat.extend(data[:,1]); all_z_flat.extend(data[:,2]); slice_indices.extend([i]*n_pts_slice)
    except Exception as e: print(f"  Warning: Skipping file {file_path.name} due to error: {e}"); continue
if not all_x_flat: print("\nError: No valid data loaded."); exit()
x_vec=np.array(all_x_flat); y_vec=np.array(all_y_flat); z_vec_original=np.array(all_z_flat); slice_indices=np.array(slice_indices)
total_points=len(x_vec); n_slices_loaded=len(np.unique(slice_indices))
print(f"\nData loaded: {total_points} points from {n_slices_loaded} slices.")

# --- Step 3: Prepare Colormap and Colors (Global + Per-Slice Logic) ---
# (Code unchanged - calculates shared_background_z_color and final point_colors)
n_colors=len(hex_colors)
try: custom_colormap_base=np.array([matplotlib.colors.to_rgb(h) for h in hex_colors])
except Exception as e: print(f"Error converting hex: {e}"); exit()
cmap_indices_base=np.linspace(0,1,n_colors); cmap_indices_interp=np.linspace(0,1,256)
interp_func_r=interp1d(cmap_indices_base,custom_colormap_base[:,0]); interp_func_g=interp1d(cmap_indices_base,custom_colormap_base[:,1]); interp_func_b=interp1d(cmap_indices_base,custom_colormap_base[:,2])
colormap_interp=np.vstack([interp_func_r(cmap_indices_interp),interp_func_g(cmap_indices_interp),interp_func_b(cmap_indices_interp)]).T
print("Colormap prepared.")
print(f"\nDetermining shared color for Z={z_target_value}...")
global_z_min=np.min(z_vec_original); global_z_max=np.max(z_vec_original)
print(f"  Global Z range: [{global_z_min:.4f}, {global_z_max:.4f}]")
if global_z_max > global_z_min: norm_target_z=(z_target_value-global_z_min)/(global_z_max-global_z_min); norm_target_z=np.clip(norm_target_z,0.,1.); target_z_color_idx=np.floor(norm_target_z*255.999).astype(int); shared_background_z_color=colormap_interp[target_z_color_idx,:]; print(f"  Shared BG/Override Color: {shared_background_z_color}")
else: print(f"  Warning: Global Z range zero."); shared_background_z_color=colormap_interp[127,:]
colors_rgb_per_slice=np.zeros((total_points,3))
print("\nCalculating per-slice base colors...")
unique_slice_ids=np.unique(slice_indices)
for slice_id in unique_slice_ids: mask=(slice_indices==slice_id); z_col=z_vec_original[mask]; z_min,z_max=np.min(z_col),np.max(z_col); z_norm=(z_col-z_min)/(z_max-z_min) if z_max>z_min else np.zeros(z_col.shape); z_idx=np.floor(z_norm*255.999).astype(int); z_idx=np.clip(z_idx,0,255); colors_rgb_per_slice[mask,:]=colormap_interp[z_idx,:]
print("Per-slice calculation complete.")
print(f"Applying override for Z ≈ {z_target_value}...")
is_target_z=np.isclose(z_vec_original,z_target_value,atol=z_mask_tolerance); num_z_override=np.sum(is_target_z); print(f"  Overriding {num_z_override} points.")
point_colors=np.where(is_target_z[:,np.newaxis],np.array(shared_background_z_color).reshape(1,3),colors_rgb_per_slice)
print("Final point colors determined.")

# --- Step 4: Create Verification Scatter Plot ---
# (Uses reversed plotting order - code unchanged)
print(f"\nCreating verification scatter plot ({output_filename_scatter})...")
try:
    fig,ax=plt.subplots(figsize=(10,8)); marker_size=1
    ax.scatter(x_vec[::-1],y_vec[::-1],s=marker_size,c=point_colors[::-1],marker='o',edgecolors='none',rasterized=True)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('Scatter Plot (Final Colors, Reversed Plot Order)')
    ax.axis('equal'); ax.set_xlim(x_limits_fixed); ax.set_ylim(y_limits_fixed); ax.grid(True,ls=':',alpha=0.6); ax.set_facecolor((0.98,0.98,0.98))
    norm=matplotlib.colors.Normalize(vmin=0,vmax=1); sm=matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(colormap_interp),norm=norm); sm.set_array([])
    fig.colorbar(sm,ax=ax,label='Norm. Z (Per-Slice Basis)',shrink=0.75)
    fig.savefig(output_filename_scatter,dpi=150,bbox_inches='tight'); print(f"Scatter plot saved.")
except Exception as e: print(f"Warning: Scatter plot failed: {e}")
finally: plt.close(fig) if 'fig' in locals() else None

# --- Step 5: Create Image by Direct Pixel Mapping (Reversed Order) ---
print(f'\nCreating direct map image ({output_filename_direct_rev})...')

# Initialize with the SHARED calculated background color
final_image_direct = np.ones((image_height_pixels, image_width_pixels, 3), dtype=float) * np.array(shared_background_z_color).reshape(1, 1, 3)
print(f"Initialized image with shared background color.")

# Calculate target pixel coordinates (centers)
print("Mapping data points to pixel coordinates...")
px = np.round((x_vec - x_limits_fixed[0]) / (x_limits_fixed[1] - x_limits_fixed[0]) * (image_width_pixels - 1)).astype(int)
py = np.round((y_vec - y_limits_fixed[0]) / (y_limits_fixed[1] - y_limits_fixed[0]) * (image_height_pixels - 1)).astype(int)
px = np.clip(px, 0, image_width_pixels - 1); py = np.clip(py, 0, image_height_pixels - 1)
print("Pixel coordinates calculated.")

# Assign BASE colors using REVERSED order to combat obscuration at center pixel
px_rev = px[::-1]
py_rev = py[::-1]
point_colors_rev = point_colors[::-1] # Use final colors (with override) reversed
print(f"Assigning base pixel colors using reversed order...")
final_image_direct[py_rev, px_rev, :] = point_colors_rev
print("Base pixel colors assigned.")

# --- Apply Neighborhood Thickening (if radius > 0) with REVERSED LOOP ORDER ---
if point_radius > 0:
    print(f"\nApplying neighborhood thickening (radius={point_radius}) with reversed loop...")
    # Step 1: Apply neighborhood thickening (as before, reversed order)
    #for i in range(total_points - 1, -1, -1):  # reversed order
    for i in range(total_points):
        center_py, center_px = py[i], px[i]
        color = point_colors[i]
        row_start = max(0, center_py - point_radius)
        row_end = min(image_height_pixels, center_py + point_radius + 1)
        col_start = max(0, center_px - point_radius)
        col_end = min(image_width_pixels, center_px + point_radius + 1)
        final_image_direct[row_start:row_end, col_start:col_end, :] = color

    # Step 2: Re-assign exact center pixels to guarantee their visibility
    print("  Repainting center pixels to restore detail...")
    for i in range(total_points):
        final_image_direct[py[i], px[i], :] = point_colors[i]

    print("Neighborhood thickening + center point correction complete.")
else:
    print("\nSkipping neighborhood thickening (point_radius=0).")



# --- Step 6: Final Flip and Save ---
# (Code unchanged - saves final_image_direct)
print("\nApplying final vertical flip (np.flipud).")
final_image_flipped = np.flipud(final_image_direct)
print(f'Final image constructed ({final_image_flipped.shape}) and flipped.')
try:
    output_path = Path(output_filename_direct_rev); print(f'\nSaving final image as {output_path.name}...')
    final_image_save = (np.clip(final_image_flipped, 0.0, 1.0) * 255).astype(np.uint8)
    if final_image_save.shape != (image_height_pixels, image_width_pixels, 3): print(f"WARNING: Shape mismatch {final_image_save.shape}!")
    imageio.imwrite(output_path, final_image_save); print(f'Image saved successfully.')
except ImportError: print("\nERROR: imageio not found.")
except Exception as e: print(f"\nFATAL ERROR saving image: {e}")

print("\nScript finished.")