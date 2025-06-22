import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_nifti(path):
    """
    Load a NIfTI (.nii or .nii.gz) file and return numpy array.
    path: path to the NIfTI file
    Returns:
        data: numpy array of the image data
    """
    img = nib.load(path)
    data = img.get_fdata()
    return data

def match_ct_and_mask(ct_path, mask_path):
    """
    Load CT and mask NIfTI files, check shape match, and return data.
    ct_path: path to CT NIfTI file
    mask_path: path to mask NIfTI file
    Returns:
        ct: numpy array of CT data
        mask: numpy array of mask data
        indices: list of slice indices with non-zero mask pixels
    """
    ct = load_nifti(ct_path)
    mask = load_nifti(mask_path)
    print(f"CT shape: {ct.shape}, Mask shape: {mask.shape}")
    assert ct.shape == mask.shape, f"Shape mismatch: CT {ct.shape} vs mask {mask.shape}"
    indices = get_nonzero_slices(mask)
    return ct, mask, indices

def normalize_ct(ct, min_hu=-200, max_hu=250):
    """
    Normalize CT volume or slice to [0,1] in HU window.
    Accepts 2D or 3D np.array.
    ct: numpy array of shape (H, W) or (H, W, num_slices)
    min_hu: minimum HU value for normalization (default -200)
    max_hu: maximum HU value for normalization (default 250)
    Returns:
        norm: numpy array of the same shape as ct, normalized to [0, 1]
    """
    ct_clip = np.clip(ct, min_hu, max_hu)
    norm = (ct_clip - min_hu) / (max_hu - min_hu)
    return norm

def get_nonzero_slices(mask, threshold=0):
    """
    Return indices of slices in which the mask has non-zero pixels (for 3D volume).
    mask: numpy array of shape (H, W, num_slices)
    threshold: minimum pixel value to consider as non-zero (default 0)
    Returns:
        indices: list of slice indices where mask has non-zero pixels
    """
    nonzero = np.any(mask > threshold, axis=(0, 1))
    indices = np.where(nonzero)[0]
    return indices

def get_largest_tumor_slice(mask):
    """
    Return index of the slice with the largest mask area (max tumor pixels).
    mask: numpy array of shape (H, W, num_slices)
    Returns:
        max_idx: index of the slice with the largest area
    """
    slice_areas = [np.sum(mask[:,:,i] > 0) for i in range(mask.shape[2])]
    max_idx = int(np.argmax(slice_areas))
    return max_idx

def crop_roi(ct_slice, mask_slice, margin=10, out_size=(256, 256)):
    """
    Crop region of interest (ROI) around the tumor in the CT slice.
    ct_slice: 2D numpy array of shape (H, W) for CT slice
    mask_slice: 2D numpy array of shape (H, W) for mask slice
    margin: margin around the tumor to crop (default 10 pixels)
    out_size: output size for the cropped patches (H, W)
    Returns:
        roi_ct: cropped and resized CT slice
        roi_mask: cropped and resized mask slice
    """
    y_idx, x_idx = np.where(mask_slice > 0)
    if len(y_idx) == 0 or len(x_idx) == 0:
        roi_ct = cv2.resize(ct_slice, out_size)
        roi_mask = cv2.resize(mask_slice, out_size)
        return roi_ct, roi_mask
    y_min, y_max = max(y_idx.min() - margin, 0), min(y_idx.max() + margin, ct_slice.shape[0])
    x_min, x_max = max(x_idx.min() - margin, 0), min(x_idx.max() + margin, ct_slice.shape[1])
    roi_ct = ct_slice[y_min:y_max, x_min:x_max]
    roi_mask = mask_slice[y_min:y_max, x_min:x_max]
    roi_ct = cv2.resize(roi_ct, out_size, interpolation=cv2.INTER_LINEAR)
    roi_mask = cv2.resize(roi_mask, out_size, interpolation=cv2.INTER_NEAREST)
    return roi_ct, roi_mask

def get_processed_slice(ct, mask, slice_idx, margin=10, out_size=(256, 256)):
    """
    Get preprocessed CT/mask slice (normalized, cropped, resized).
    ct: 3D numpy array of shape (H, W, num_slices)
    mask: 3D numpy array of shape (H, W, num_slices)
    slice_idx: index of the slice to process
    margin: margin around the tumor to crop
    out_size: output size for the cropped patches (H, W)
    Returns:
        stacked: numpy array of shape (2, H, W) with [CT slice, mask slice]
    """
    ct_slice = ct[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]
    ct_norm = normalize_ct(ct_slice)
    # print(f"Processing slice {slice_idx}: CT shape {ct_norm.shape}, Mask shape {mask_slice.shape}")
    roi_ct, roi_mask = crop_roi(ct_norm, mask_slice, margin=margin, out_size=out_size)
    # print(f"Cropped ROI CT shape: {roi_ct.shape}, Mask shape: {roi_mask.shape}")
    stacked = np.stack([roi_ct, roi_mask], axis=0)
    print(f"Stacked shape: {stacked.shape}")
    return stacked

# Optional: get the slice with the largest tumor area
def get_processed_largest_slice(ct, mask, margin=10, out_size=(256, 256)):
    """
    Get preprocessed slice with the largest tumor area.
    ct: 3D numpy array of shape (H, W, num_slices)
    mask: 3D numpy array of shape (H, W, num_slices)
    margin: margin around the tumor to crop
    out_size: output size for the cropped patches (H, W)
    Returns:
        stacked: numpy array of shape (2, H, W) with [CT slice, mask slice]
        idx: index of the slice with the largest tumor area
    """
    idx = get_largest_tumor_slice(mask)
    return get_processed_slice(ct, mask, idx, margin=margin, out_size=out_size), idx

def get_processed_all_tumor_slices(ct, mask, margin=10, out_size=(256, 256)):
    """
    Obtain all tumor slices with non-zero mask pixels.
    Returns a list of tuples (processed_slice, slice_index).
    Each processed_slice is a numpy array of shape (2, H, W).
    mask: 3D numpy array of shape (H, W, num_slices)
    ct: 3D numpy array of shape (H, W, num_slices)
    margin: margin around the tumor to crop.
    out_size: output size for the cropped patches (H, W)
    Returns:
        results: list of tuples (processed_slice, slice_index)
    """
    indices = get_nonzero_slices(mask)
    results = []
    for idx in indices:
        processed = get_processed_slice(ct, mask, idx, margin=margin, out_size=out_size)
        if idx % 50 == 0:
            visualize_slice_with_mask(ct, mask, idx)
        results.append((processed, idx))
    print(f"Found {len(results)} tumor slices with non-zero mask pixels.")
    return results

def get_processed_2_5d_slices(ct, mask, center_idx, N=5, margin=10, out_size=(256, 256)):
    """
    Generate 2.5D slices around a center slice index.
    Returns a stack of slices centered at center_idx, with N slices in total.
    ct: 3D numpy array of shape (H, W, num_slices)
    mask: 3D numpy array of shape (H, W, num_slices)
    center_idx: index of the center slice
    N: total number of slices to generate (must be odd)
    margin: margin around the tumor to crop
    out_size: output size for the cropped patches (H, W)
    Returns:
        arr: numpy array of shape (2, N, H, W) with [CT slices, mask slices]
    """
    assert N % 2 == 1, "N must be odd"
    half = N // 2
    slices = []
    for offset in range(-half, half+1):
        idx = center_idx + offset
        if idx < 0 or idx >= ct.shape[2]:
            ct_slice = np.zeros_like(ct[:, :, 0])
            mask_slice = np.zeros_like(mask[:, :, 0])
        else:
            ct_slice = ct[:, :, idx]
            mask_slice = mask[:, :, idx]
        ct_norm = normalize_ct(ct_slice)
        roi_ct, roi_mask = crop_roi(ct_norm, mask_slice, margin=margin, out_size=out_size)
        slices.append([roi_ct, roi_mask])
    arr = np.stack(slices, axis=1)  # [2, N, H, W]
    return arr



# Optional: get a 3D patch around the tumor
def get_processed_3d_patch(ct, mask, margin=10, out_size=(128, 128, 64)):
    """
    （可选）按全3D crop的方式处理，返回标准patch大小，适用于3D网络。
    注意：这里默认全体肿瘤范围内crop，超出范围则pad或resize。
    """
    y_idx, x_idx, z_idx = np.where(mask > 0)
    if len(y_idx) == 0 or len(x_idx) == 0 or len(z_idx) == 0:
        # 没肿瘤直接resize整volume
        ct_norm = normalize_ct(ct)
        ct_3d = cv2.resize(ct_norm, out_size[:2])
        mask_3d = cv2.resize(mask, out_size[:2])
        ct_3d = np.expand_dims(ct_3d, axis=-1)
        mask_3d = np.expand_dims(mask_3d, axis=-1)
        # 这里只做了2D resize，可根据实际加3D resize包（如 scipy.ndimage.zoom）
        return np.stack([ct_3d, mask_3d], axis=0)
    y_min, y_max = max(y_idx.min() - margin, 0), min(y_idx.max() + margin, ct.shape[0])
    x_min, x_max = max(x_idx.min() - margin, 0), min(x_idx.max() + margin, ct.shape[1])
    z_min, z_max = max(z_idx.min() - margin, 0), min(z_idx.max() + margin, ct.shape[2])
    ct_crop = ct[y_min:y_max, x_min:x_max, z_min:z_max]
    mask_crop = mask[y_min:y_max, x_min:x_max, z_min:z_max]
    ct_crop = normalize_ct(ct_crop)
    # 3D resize: 用scipy的zoom或者其他方法
    try:
        from scipy.ndimage import zoom
        factors = (out_size[0] / ct_crop.shape[0], out_size[1] / ct_crop.shape[1], out_size[2] / ct_crop.shape[2])
        ct_resize = zoom(ct_crop, factors, order=1)
        mask_resize = zoom(mask_crop, factors, order=0)
        return np.stack([ct_resize, mask_resize], axis=0)
    except ImportError:
        print("Scipy not installed, 3D resize skipped.")
        return np.stack([ct_crop, mask_crop], axis=0)

# Optional: visualize a single slice with mask overlay
def visualize_slice_with_mask(ct, mask, slice_idx, figsize=(6,6), save_path=None):
    """
    可视化ct+mask叠加（可显示也可保存）。
    """
    plt.figure(figsize=figsize)
    plt.imshow(ct[:,:,slice_idx], cmap='gray')
    plt.imshow(mask[:,:,slice_idx], cmap='Reds', alpha=0.4)
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

