import os
import glob
import argparse
import json
import nibabel as nib
import numpy as np
from collections import defaultdict
from datetime import datetime


def get_file_pairs(data_dir):
    """Get matched image-mask pairs."""
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
    
    image_basenames = {os.path.basename(f).replace('.nii.gz', '') for f in image_files}
    mask_basenames = {os.path.basename(f).replace('.nii.gz', '') for f in mask_files}
    
    matched = image_basenames & mask_basenames
    only_images = image_basenames - mask_basenames
    only_masks = mask_basenames - image_basenames
    
    matched_files = sorted([f for f in image_files 
                           if os.path.basename(f).replace('.nii.gz', '') in matched])
    
    return image_dir, mask_dir, image_files, mask_files, matched, only_images, only_masks, matched_files


def process_case(img_path, mask_dir):
    """Process a single image-mask pair and extract statistics."""
    basename = os.path.basename(img_path).replace('.nii.gz', '')
    mask_path = os.path.join(mask_dir, f"{basename}.nii.gz")
    
    try:
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        
        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.uint8)
        
        if img_data.shape != mask_data.shape:
            return None, f"{basename}: shape mismatch {img_data.shape} vs {mask_data.shape}"
        
        spacing = img_nii.header.get_zooms()[:3]
        
        unique_labels, counts = np.unique(mask_data, return_counts=True)
        label_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        
        foreground_ratio = (mask_data > 0).sum() / mask_data.size
        
        roi_size = None
        if foreground_ratio > 0:
            coords = np.argwhere(mask_data > 0)
            if len(coords) > 0:
                minz, miny, minx = coords.min(axis=0)
                maxz, maxy, maxx = coords.max(axis=0)
                roi_size = (maxx - minx + 1, maxy - miny + 1, maxz - minz + 1)
        
        stats = {
            'shape': img_data.shape,
            'spacing': spacing,
            'intensity_min': float(img_data.min()),
            'intensity_max': float(img_data.max()),
            'intensity_mean': float(img_data.mean()),
            'label_counts': label_counts,
            'foreground_ratio': foreground_ratio,
            'roi_size': roi_size,
            'dtype': str(img_data.dtype),
        }
        
        return stats, None
        
    except Exception as e:
        return None, f"{basename}: {str(e)}"


def print_statistics(shapes, spacings, intensity_mins, intensity_maxs, intensity_means,
                    label_values, foreground_ratios, roi_sizes, dtypes, errors, verbose):
    """Print all statistics."""
    print(f"\n[3] Shape Statistics:")
    print(f"    Total cases: {len(shapes)}")
    print(f"    X: {shapes[:, 0].min()} - {shapes[:, 0].max()} (median: {np.median(shapes[:, 0]):.0f})")
    print(f"    Y: {shapes[:, 1].min()} - {shapes[:, 1].max()} (median: {np.median(shapes[:, 1]):.0f})")
    print(f"    Z: {shapes[:, 2].min()} - {shapes[:, 2].max()} (median: {np.median(shapes[:, 2]):.0f})")
    
    unique_shapes = {tuple(s) for s in shapes}
    if len(unique_shapes) > 1:
        print(f"    ⚠️  Multiple shapes: {len(unique_shapes)} unique")
        if verbose:
            shape_counts = defaultdict(int)
            for s in shapes:
                shape_counts[tuple(s)] += 1
            print("    Most common:")
            for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {shape}: {count} cases")
    
    print(f"\n[4] Spacing Statistics (mm):")
    print(f"    X: {spacings[:, 0].min():.3f} - {spacings[:, 0].max():.3f} (median: {np.median(spacings[:, 0]):.3f})")
    print(f"    Y: {spacings[:, 1].min():.3f} - {spacings[:, 1].max():.3f} (median: {np.median(spacings[:, 1]):.3f})")
    print(f"    Z: {spacings[:, 2].min():.3f} - {spacings[:, 2].max():.3f} (median: {np.median(spacings[:, 2]):.3f})")
    
    unique_spacings = {tuple(s) for s in spacings}
    spacing_counts = None
    if len(unique_spacings) > 1:
        print(f"    ⚠️  Multiple spacings: {len(unique_spacings)} unique")
        if verbose:
            spacing_counts = defaultdict(int)
            for s in spacings:
                spacing_counts[tuple(s)] += 1
            print("    Most common:")
            for spacing, count in sorted(spacing_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {spacing}: {count} cases")
    
    print(f"\n[5] Intensity Statistics (HU):")
    print(f"    Global: {intensity_mins.min():.1f} - {intensity_maxs.max():.1f}")
    print(f"    Median: {np.median(intensity_mins):.1f} - {np.median(intensity_maxs):.1f}")
    print(f"    Mean: {intensity_means.mean():.1f} ± {intensity_means.std():.1f}")
    a_min = int(np.percentile(intensity_mins, 5))
    a_max = int(np.percentile(intensity_maxs, 95))
    print(f"    Recommended normalization: a_min={a_min}, a_max={a_max}")
    
    print(f"\n[6] Label Statistics:")
    total_voxels = sum(label_values.values())
    print(f"    Labels found: {sorted(label_values.keys())}")
    for label in sorted(label_values.keys()):
        ratio = label_values[label] / total_voxels
        print(f"    Label {label}: {label_values[label]:,} voxels ({ratio*100:.2f}%)")
    
    print(f"\n[7] Foreground Statistics:")
    print(f"    Mean: {foreground_ratios.mean():.6f}")
    print(f"    Range: {foreground_ratios.min():.6f} - {foreground_ratios.max():.6f}")
    print(f"    Median: {np.median(foreground_ratios):.6f}")
    empty_masks = (foreground_ratios == 0).sum()
    if empty_masks > 0:
        print(f"    ⚠️  Empty masks: {empty_masks} cases")
    
    if roi_sizes:
        roi_sizes_arr = np.array(roi_sizes)
        print(f"\n[8] ROI Size Statistics:")
        print(f"    X: {roi_sizes_arr[:, 0].min()} - {roi_sizes_arr[:, 0].max()} (median: {np.median(roi_sizes_arr[:, 0]):.0f})")
        print(f"    Y: {roi_sizes_arr[:, 1].min()} - {roi_sizes_arr[:, 1].max()} (median: {np.median(roi_sizes_arr[:, 1]):.0f})")
        print(f"    Z: {roi_sizes_arr[:, 2].min()} - {roi_sizes_arr[:, 2].max()} (median: {np.median(roi_sizes_arr[:, 2]):.0f})")
    
    print(f"\n[9] Data Types:")
    dtype_counts = defaultdict(int)
    for dt in dtypes:
        dtype_counts[dt] += 1
    for dt, count in dtype_counts.items():
        print(f"    {dt}: {count} cases")
    
    return spacing_counts, a_min, a_max


def print_recommendations(shapes, roi_sizes, spacing_counts, a_min, a_max):
    """Print model configuration recommendations."""
    print(f"\n[10] Model Configuration Recommendations:")
    
    median_shape = np.median(shapes, axis=0).astype(int)
    patch_x = min(160, int(median_shape[0] * 0.5))
    patch_y = min(160, int(median_shape[1] * 0.5))
    patch_z = min(80, int(median_shape[2] * 0.3))
    recommended_patch = (patch_x, patch_y, patch_z)
    
    print(f"    Recommended patch_size: {recommended_patch}")
    print(f"    Reason: ~50% of median image size")
    
    if roi_sizes:
        median_roi = np.median(np.array(roi_sizes), axis=0).astype(int)
        print(f"    Median ROI size: {tuple(median_roi)}")
        if any(roi > patch for roi, patch in zip(median_roi, recommended_patch)):
            print(f"    ⚠️  Some ROIs larger than recommended patch size")
    
    patch_memory_mb = np.prod(recommended_patch) * 4 * 2 / (1024**2)
    print(f"    Memory per patch: ~{patch_memory_mb:.1f} MB")
    
    for batch_size in [4, 8, 16]:
        batch_memory_gb = patch_memory_mb * batch_size / 1024
        print(f"    Batch size {batch_size}: ~{batch_memory_gb:.2f} GB")
    
    print(f"\n    Recommended normalization:")
    print(f"        ScaleIntensityRanged(keys=['image'], a_min={a_min}, a_max={a_max}, b_min=0.0, b_max=1.0, clip=True)")
    
    if spacing_counts:
        most_common_spacing = max(spacing_counts.items(), key=lambda x: x[1])[0]
        print(f"\n    ⚠️  Consider using Spacingd transform")
        print(f"        Most common spacing: {most_common_spacing}")
    
    return recommended_patch


def build_results(shapes, spacings, intensity_mins, intensity_maxs, intensity_means,
                 foreground_ratios, label_values, roi_sizes, matched, errors,
                 recommended_patch, a_min, a_max):
    """Build results dictionary for JSON export."""
    results = {
        'num_cases': len(shapes),
        'num_matched_pairs': len(matched),
        'num_errors': len(errors),
        'shape_stats': {
            'min': shapes.min(axis=0).tolist(),
            'max': shapes.max(axis=0).tolist(),
            'median': np.median(shapes, axis=0).tolist(),
        },
        'spacing_stats': {
            'min': spacings.min(axis=0).tolist(),
            'max': spacings.max(axis=0).tolist(),
            'median': np.median(spacings, axis=0).tolist(),
        },
        'intensity_range': {
            'global_min': float(intensity_mins.min()),
            'global_max': float(intensity_maxs.max()),
            'median_min': float(np.median(intensity_mins)),
            'median_max': float(np.median(intensity_maxs)),
        },
        'foreground_ratio': {
            'mean': float(foreground_ratios.mean()),
            'min': float(foreground_ratios.min()),
            'max': float(foreground_ratios.max()),
            'median': float(np.median(foreground_ratios)),
        },
        'label_distribution': {str(k): int(v) for k, v in label_values.items()},
        'recommended_patch_size': recommended_patch,
        'recommended_normalization': {'a_min': a_min, 'a_max': a_max},
        'check_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if roi_sizes:
        roi_sizes_arr = np.array(roi_sizes)
        results['roi_size_stats'] = {
            'min': roi_sizes_arr.min(axis=0).tolist(),
            'max': roi_sizes_arr.max(axis=0).tolist(),
            'median': np.median(roi_sizes_arr, axis=0).tolist(),
        }
    
    return results


def check_dataset(data_dir="./data/raw", verbose=True):
    """Comprehensive dataset check for 3D medical image segmentation."""
    image_dir, mask_dir, image_files, mask_files, matched, only_images, only_masks, matched_files = get_file_pairs(data_dir)
    
    print("=" * 80)
    print("DATASET COMPREHENSIVE CHECK")
    print("=" * 80)
    print(f"\nImage directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Found {len(image_files)} images, {len(mask_files)} masks")
    
    if len(image_files) == 0 or len(mask_files) == 0:
        print("ERROR: No files found!")
        return None
    
    print(f"\n[1] File Matching:")
    print(f"    Matched pairs: {len(matched)}")
    if only_images:
        print(f"    ⚠️  Images without masks: {len(only_images)}")
        if verbose and len(only_images) <= 10:
            print(f"       {list(only_images)[:10]}")
    if only_masks:
        print(f"    ⚠️  Masks without images: {len(only_masks)}")
        if verbose and len(only_masks) <= 10:
            print(f"       {list(only_masks)[:10]}")
    
    if len(matched) == 0:
        print("ERROR: No matched pairs found!")
        return None
    
    shapes = []
    spacings = []
    intensity_mins = []
    intensity_maxs = []
    intensity_means = []
    label_values = defaultdict(int)
    roi_sizes = []
    foreground_ratios = []
    dtypes = []
    errors = []
    
    print(f"\n[2] Processing {len(matched_files)} matched pairs...")
    for i, img_path in enumerate(matched_files):
        if verbose and (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(matched_files)}...")
        
        stats, error = process_case(img_path, mask_dir)
        if error:
            errors.append(error)
            continue
        
        shapes.append(stats['shape'])
        spacings.append(stats['spacing'])
        intensity_mins.append(stats['intensity_min'])
        intensity_maxs.append(stats['intensity_max'])
        intensity_means.append(stats['intensity_mean'])
        foreground_ratios.append(stats['foreground_ratio'])
        dtypes.append(stats['dtype'])
        
        for label, count in stats['label_counts'].items():
            label_values[label] += count
        
        if stats['roi_size']:
            roi_sizes.append(stats['roi_size'])
    
    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for err in errors[:10]:
            print(f"    {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
    
    if len(shapes) == 0:
        print("ERROR: No valid cases processed!")
        return None
    
    shapes = np.array(shapes)
    spacings = np.array(spacings)
    intensity_mins = np.array(intensity_mins)
    intensity_maxs = np.array(intensity_maxs)
    intensity_means = np.array(intensity_means)
    foreground_ratios = np.array(foreground_ratios)
    
    spacing_counts, a_min, a_max = print_statistics(
        shapes, spacings, intensity_mins, intensity_maxs, intensity_means,
        label_values, foreground_ratios, roi_sizes, dtypes, errors, verbose
    )
    
    recommended_patch = print_recommendations(shapes, roi_sizes, spacing_counts, a_min, a_max)
    
    print("\n" + "=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)
    
    results = build_results(
        shapes, spacings, intensity_mins, intensity_maxs, intensity_means,
        foreground_ratios, label_values, roi_sizes, matched, errors,
        recommended_patch, a_min, a_max
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive dataset check")
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Data directory")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--save_json", type=str, default="", help="Custom JSON save path")
    parser.add_argument("--no_save", action="store_true", help="Don't save to file")
    args = parser.parse_args()
    
    results = check_dataset(args.data_dir, verbose=not args.quiet)
    
    if results and not args.no_save:
        if args.save_json:
            save_path = args.save_json
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"data/dataset_info_{timestamp}.json"
        
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        abs_path = os.path.abspath(save_path)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to: {abs_path}")
