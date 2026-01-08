import os
import glob
import argparse
import json
import nibabel as nib
import numpy as np
from collections import defaultdict
from datetime import datetime


def check_dataset(data_dir="./data/raw", verbose=True):
    """
    Comprehensive dataset check for 3D medical image segmentation.
    
    Checks:
    - File matching (images vs masks)
    - Shape consistency
    - Spacing distribution
    - Intensity range (HU values)
    - Label values and class distribution
    - ROI size statistics
    - Data type and memory usage
    - Recommendations for model configuration
    """
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
    
    print("=" * 80)
    print("DATASET COMPREHENSIVE CHECK")
    print("=" * 80)
    print(f"\nImage directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Found {len(image_files)} images, {len(mask_files)} masks")
    
    if len(image_files) == 0 or len(mask_files) == 0:
        print("ERROR: No files found!")
        return
    
    image_basenames = {os.path.basename(f).replace('.nii.gz', '') for f in image_files}
    mask_basenames = {os.path.basename(f).replace('.nii.gz', '') for f in mask_files}
    
    matched = image_basenames & mask_basenames
    only_images = image_basenames - mask_basenames
    only_masks = mask_basenames - image_basenames
    
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
        return
    
    matched_files = sorted([f for f in image_files if os.path.basename(f).replace('.nii.gz', '') in matched])
    
    shapes = []
    spacings = []
    intensity_mins = []
    intensity_maxs = []
    intensity_means = []
    mask_shapes = []
    label_values = defaultdict(int)
    roi_sizes = []
    foreground_ratios = []
    dtypes = []
    
    errors = []
    
    print(f"\n[2] Processing {len(matched_files)} matched pairs...")
    for i, img_path in enumerate(matched_files):
        if verbose and (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(matched_files)}...")
        
        basename = os.path.basename(img_path).replace('.nii.gz', '')
        mask_path = os.path.join(mask_dir, f"{basename}.nii.gz")
        
        try:
            img_nii = nib.load(img_path)
            mask_nii = nib.load(mask_path)
            
            img_data = img_nii.get_fdata().astype(np.float32)
            mask_data = mask_nii.get_fdata().astype(np.uint8)
            
            img_shape = img_data.shape
            mask_shape = mask_data.shape
            
            if img_shape != mask_shape:
                errors.append(f"{basename}: shape mismatch {img_shape} vs {mask_shape}")
                continue
            
            shapes.append(img_shape)
            mask_shapes.append(mask_shape)
            
            spacing = img_nii.header.get_zooms()[:3]
            spacings.append(spacing)
            
            intensity_mins.append(img_data.min())
            intensity_maxs.append(img_data.max())
            intensity_means.append(img_data.mean())
            
            unique_labels, counts = np.unique(mask_data, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_values[int(label)] += count
            
            foreground_ratio = (mask_data > 0).sum() / mask_data.size
            foreground_ratios.append(foreground_ratio)
            
            if foreground_ratio > 0:
                coords = np.argwhere(mask_data > 0)
                if len(coords) > 0:
                    minz, miny, minx = coords.min(axis=0)
                    maxz, maxy, maxx = coords.max(axis=0)
                    roi_size = (maxx - minx + 1, maxy - miny + 1, maxz - minz + 1)
                    roi_sizes.append(roi_size)
            
            dtypes.append(str(img_data.dtype))
            
        except Exception as e:
            errors.append(f"{basename}: {str(e)}")
    
    if errors:
        print(f"\n⚠️  Errors encountered ({len(errors)}):")
        for err in errors[:10]:
            print(f"    {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
    
    shapes = np.array(shapes)
    spacings = np.array(spacings)
    intensity_mins = np.array(intensity_mins)
    intensity_maxs = np.array(intensity_maxs)
    intensity_means = np.array(intensity_means)
    foreground_ratios = np.array(foreground_ratios)
    
    print(f"\n[3] Shape Statistics:")
    print(f"    Total cases: {len(shapes)}")
    print(f"    Shape range (X): {shapes[:, 0].min()} - {shapes[:, 0].max()} (median: {np.median(shapes[:, 0]):.0f})")
    print(f"    Shape range (Y): {shapes[:, 1].min()} - {shapes[:, 1].max()} (median: {np.median(shapes[:, 1]):.0f})")
    print(f"    Shape range (Z): {shapes[:, 2].min()} - {shapes[:, 2].max()} (median: {np.median(shapes[:, 2]):.0f})")
    
    unique_shapes = {tuple(s) for s in shapes}
    if len(unique_shapes) > 1:
        print(f"    ⚠️  Multiple shapes detected: {len(unique_shapes)} unique shapes")
        if verbose:
            shape_counts = defaultdict(int)
            for s in shapes:
                shape_counts[tuple(s)] += 1
            print(f"    Most common shapes:")
            for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {shape}: {count} cases")
    
    print(f"\n[4] Spacing Statistics (mm):")
    print(f"    X spacing: {spacings[:, 0].min():.3f} - {spacings[:, 0].max():.3f} (median: {np.median(spacings[:, 0]):.3f})")
    print(f"    Y spacing: {spacings[:, 1].min():.3f} - {spacings[:, 1].max():.3f} (median: {np.median(spacings[:, 1]):.3f})")
    print(f"    Z spacing: {spacings[:, 2].min():.3f} - {spacings[:, 2].max():.3f} (median: {np.median(spacings[:, 2]):.3f})")
    
    unique_spacings = {tuple(s) for s in spacings}
    if len(unique_spacings) > 1:
        print(f"    ⚠️  Multiple spacings detected: {len(unique_spacings)} unique spacings")
        if verbose:
            spacing_counts = defaultdict(int)
            for s in spacings:
                spacing_counts[tuple(s)] += 1
            print(f"    Most common spacings:")
            for spacing, count in sorted(spacing_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {spacing}: {count} cases")
    
    print(f"\n[5] Intensity Statistics (HU values):")
    print(f"    Global min: {intensity_mins.min():.1f}, max: {intensity_maxs.max():.1f}")
    print(f"    Median min: {np.median(intensity_mins):.1f}, median max: {np.median(intensity_maxs):.1f}")
    print(f"    Mean intensity: {intensity_means.mean():.1f} ± {intensity_means.std():.1f}")
    print(f"    Recommended normalization: a_min={int(np.percentile(intensity_mins, 5)):.0f}, "
          f"a_max={int(np.percentile(intensity_maxs, 95)):.0f}")
    
    print(f"\n[6] Label Statistics:")
    total_voxels = sum(label_values.values())
    print(f"    Label values found: {sorted(label_values.keys())}")
    for label in sorted(label_values.keys()):
        count = label_values[label]
        ratio = count / total_voxels
        print(f"    Label {label}: {count:,} voxels ({ratio*100:.2f}%)")
    
    print(f"\n[7] Foreground Statistics:")
    print(f"    Mean foreground ratio: {foreground_ratios.mean():.6f}")
    print(f"    Min: {foreground_ratios.min():.6f}, Max: {foreground_ratios.max():.6f}")
    print(f"    Median: {np.median(foreground_ratios):.6f}")
    empty_masks = (foreground_ratios == 0).sum()
    if empty_masks > 0:
        print(f"    ⚠️  Empty masks: {empty_masks} cases")
    
    if roi_sizes:
        roi_sizes = np.array(roi_sizes)
        print(f"\n[8] ROI Size Statistics (bounding box of foreground):")
        print(f"    X size: {roi_sizes[:, 0].min()} - {roi_sizes[:, 0].max()} (median: {np.median(roi_sizes[:, 0]):.0f})")
        print(f"    Y size: {roi_sizes[:, 1].min()} - {roi_sizes[:, 1].max()} (median: {np.median(roi_sizes[:, 1]):.0f})")
        print(f"    Z size: {roi_sizes[:, 2].min()} - {roi_sizes[:, 2].max()} (median: {np.median(roi_sizes[:, 2]):.0f})")
    
    print(f"\n[9] Data Types:")
    dtype_counts = defaultdict(int)
    for dt in dtypes:
        dtype_counts[dt] += 1
    for dt, count in dtype_counts.items():
        print(f"    {dt}: {count} cases")
    
    print(f"\n[10] Model Configuration Recommendations:")
    median_shape = np.median(shapes, axis=0).astype(int)
    median_roi = np.median(roi_sizes, axis=0).astype(int) if roi_sizes else None
    
    recommended_patch_x = min(160, int(median_shape[0] * 0.5))
    recommended_patch_y = min(160, int(median_shape[1] * 0.5))
    recommended_patch_z = min(80, int(median_shape[2] * 0.3))
    
    recommended_patch = (recommended_patch_x, recommended_patch_y, recommended_patch_z)
    
    print(f"    Recommended patch_size: {recommended_patch}")
    print(f"    Reason: ~50% of median image size, fits most GPUs")
    
    if median_roi is not None:
        print(f"    Median ROI size: {tuple(median_roi)}")
        if any(roi > patch for roi, patch in zip(median_roi, recommended_patch)):
            print(f"    ⚠️  Some ROIs are larger than recommended patch size")
    
    patch_memory_mb = np.prod(recommended_patch) * 4 * 2 / (1024**2)
    print(f"    Estimated memory per patch: ~{patch_memory_mb:.1f} MB (float32, 2 channels)")
    
    for batch_size in [4, 8, 16]:
        batch_memory_gb = patch_memory_mb * batch_size / 1024
        print(f"    Batch size {batch_size}: ~{batch_memory_gb:.2f} GB (patches only, model not included)")
    
    print(f"\n    Recommended normalization:")
    print(f"        ScaleIntensityRanged(keys=['image'], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True)")
    
    if len(unique_spacings) > 1:
        print(f"\n    ⚠️  Consider using Spacingd transform to normalize spacing")
        print(f"        Most common spacing: {max(spacing_counts.items(), key=lambda x: x[1])[0]}")
    
    print("\n" + "=" * 80)
    print("CHECK COMPLETE")
    print("=" * 80)
    
    results = {
        'num_cases': len(shapes),
        'num_matched_pairs': len(matched),
        'num_errors': len(errors),
        'shape_stats': {
            'min': shapes.min(axis=0).tolist() if len(shapes) > 0 else None,
            'max': shapes.max(axis=0).tolist() if len(shapes) > 0 else None,
            'median': np.median(shapes, axis=0).tolist() if len(shapes) > 0 else None,
        },
        'spacing_stats': {
            'min': spacings.min(axis=0).tolist() if len(spacings) > 0 else None,
            'max': spacings.max(axis=0).tolist() if len(spacings) > 0 else None,
            'median': np.median(spacings, axis=0).tolist() if len(spacings) > 0 else None,
        },
        'intensity_range': {
            'global_min': float(intensity_mins.min()) if len(intensity_mins) > 0 else None,
            'global_max': float(intensity_maxs.max()) if len(intensity_maxs) > 0 else None,
            'median_min': float(np.median(intensity_mins)) if len(intensity_mins) > 0 else None,
            'median_max': float(np.median(intensity_maxs)) if len(intensity_maxs) > 0 else None,
        },
        'foreground_ratio': {
            'mean': float(foreground_ratios.mean()) if len(foreground_ratios) > 0 else None,
            'min': float(foreground_ratios.min()) if len(foreground_ratios) > 0 else None,
            'max': float(foreground_ratios.max()) if len(foreground_ratios) > 0 else None,
            'median': float(np.median(foreground_ratios)) if len(foreground_ratios) > 0 else None,
        },
        'label_distribution': {str(k): int(v) for k, v in label_values.items()},
        'recommended_patch_size': recommended_patch,
        'recommended_normalization': {
            'a_min': int(np.percentile(intensity_mins, 5)) if len(intensity_mins) > 0 else -200,
            'a_max': int(np.percentile(intensity_maxs, 95)) if len(intensity_maxs) > 0 else 200,
        },
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive dataset check")
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Data directory")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--save_json", type=str, default="", help="Save results to JSON file (e.g., dataset_info.json)")
    args = parser.parse_args()
    
    results = check_dataset(args.data_dir, verbose=not args.quiet)
    
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to {args.save_json}")

