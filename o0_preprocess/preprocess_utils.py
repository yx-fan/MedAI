import numpy as np
import os
import nibabel as nib
import cv2
from pathlib import Path
from typing import Optional, Tuple
from scipy.ndimage import zoom, rotate

# ==== Global default params ====
DEFAULT_MARGIN = 20
DEFAULT_MIN_CROP_RATIO = 0.9


def load_nifti(path: Path):
    """Load a NIfTI file and return data, affine, header."""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header


def normalize_ct(ct: np.ndarray, min_hu: int = -200, max_hu: int = 250):
    """Clip HU values to [min_hu, max_hu] and normalize to [0, 1]."""
    ct_clip = np.clip(ct, min_hu, max_hu)
    return (ct_clip - min_hu) / (max_hu - min_hu)


def compute_global_roi(ct: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       margin: int = DEFAULT_MARGIN,
                       mode: str = "train",
                       min_crop_ratio: float = DEFAULT_MIN_CROP_RATIO):
    """
    Conservative cropping strategy:
    - train: if mask exists → crop by mask + margin
    - predict: no mask → crop center region (default 90%)
    - ensure minimum crop ratio
    """
    h, w = ct.shape[0], ct.shape[1]
    y_min, y_max, x_min, x_max = 0, h, 0, w

    if mask is not None and np.any(mask > 0):
        coords = np.column_stack(np.where(mask > 0))
        y_min, y_max = max(coords[:, 0].min() - margin, 0), min(coords[:, 0].max() + margin, h)
        x_min, x_max = max(coords[:, 1].min() - margin, 0), min(coords[:, 1].max() + margin, w)
    elif mode == "predict":
        # center crop
        crop_h, crop_w = int(h * min_crop_ratio), int(w * min_crop_ratio)
        y_min = max(0, (h - crop_h) // 2)
        y_max = y_min + crop_h
        x_min = max(0, (w - crop_w) // 2)
        x_max = x_min + crop_w

    # enforce minimum crop ratio
    crop_h, crop_w = y_max - y_min, x_max - x_min
    min_h, min_w = int(h * min_crop_ratio), int(w * min_crop_ratio)
    if crop_h < min_h:
        extra = (min_h - crop_h) // 2
        y_min = max(0, y_min - extra)
        y_max = min(h, y_max + extra)
    if crop_w < min_w:
        extra = (min_w - crop_w) // 2
        x_min = max(0, x_min - extra)
        x_max = min(w, x_max + extra)

    return y_min, y_max, x_min, x_max


def crop_roi_global(ct_slice: np.ndarray,
                    mask_slice: Optional[np.ndarray],
                    global_bbox: Tuple[int, int, int, int],
                    out_size: Tuple[int, int] = (256, 256),
                    debug_dir: Optional[str] = None,
                    debug_name: Optional[str] = None,
                    mode: str = "train"):
    """Crop single slice based on global ROI."""
    y_min, y_max, x_min, x_max = global_bbox
    ct_crop = cv2.resize(ct_slice[y_min:y_max, x_min:x_max], out_size, interpolation=cv2.INTER_LINEAR)
    mask_crop = None
    if mask_slice is not None:
        mask_crop = cv2.resize(mask_slice[y_min:y_max, x_min:x_max], out_size, interpolation=cv2.INTER_NEAREST)

    # Debug save
    if debug_dir and debug_name:
        os.makedirs(debug_dir, exist_ok=True)
        ct_vis = (ct_crop * 255).astype(np.uint8)
        if mask_crop is not None:
            mask_vis = (mask_crop > 0).astype(np.uint8) * 255
            overlay = cv2.addWeighted(ct_vis, 0.8, mask_vis, 0.2, 0)
        else:
            overlay = ct_vis
        cv2.imwrite(os.path.join(debug_dir, f"{debug_name}_{mode}.png"), overlay)

    return ct_crop, mask_crop


def augment_slice(ct_slice: np.ndarray, mask_slice: np.ndarray):
    """Data augmentation: rotation, flip, brightness change"""
    angle = np.random.uniform(-15, 15)
    ct_slice = rotate(ct_slice, angle, reshape=False, order=1, mode="nearest")
    mask_slice = rotate(mask_slice, angle, reshape=False, order=0, mode="nearest")

    if np.random.rand() > 0.5:
        ct_slice = np.fliplr(ct_slice)
        mask_slice = np.fliplr(mask_slice)

    if np.random.rand() > 0.5:
        factor = np.random.uniform(0.9, 1.1)
        ct_slice = np.clip(ct_slice * factor, 0, 1)

    return ct_slice, mask_slice


def get_processed_2d(pid, ct: np.ndarray, mask: Optional[np.ndarray], slice_idx: int,
                     global_bbox: Tuple[int, int, int, int],
                     out_size=(256, 256), mode="train", augment=False):
    """Return normalized + cropped single slice."""
    ct_norm = normalize_ct(ct[:, :, slice_idx])
    mask_slice = mask[:, :, slice_idx] if mask is not None else None
    roi_ct, roi_mask = crop_roi_global(ct_norm, mask_slice, global_bbox, out_size, mode=mode)
    if roi_mask is None:
        roi_mask = np.zeros_like(roi_ct)
    if augment and mode == "train":
        roi_ct, roi_mask = augment_slice(roi_ct, roi_mask)
    return np.stack([roi_ct, roi_mask], axis=0)


def get_processed_2_5d(pid, ct: np.ndarray, mask: Optional[np.ndarray], center_idx: int,
                       N=5, global_bbox: Tuple[int, int, int, int]=None,
                       out_size=(256, 256),
                       mode="train", augment=False):
    """Return 2.5D stack (N slices)."""
    assert N % 2 == 1, "N must be odd"
    half = N // 2
    slices = []
    for offset in range(-half, half + 1):
        idx = center_idx + offset
        if idx < 0 or idx >= ct.shape[2]:
            ct_slice = np.zeros(out_size)
            mask_slice = np.zeros(out_size)
        else:
            ct_slice = normalize_ct(ct[:, :, idx])
            mask_slice = mask[:, :, idx] if mask is not None else None
            ct_slice, mask_slice = crop_roi_global(ct_slice, mask_slice, global_bbox, out_size, mode=mode)
            if mask_slice is None:
                mask_slice = np.zeros_like(ct_slice)
            if augment and mode == "train":
                ct_slice, mask_slice = augment_slice(ct_slice, mask_slice)
        slices.append([ct_slice, mask_slice])
    return np.stack(slices, axis=1)

def get_processed_3d_patch(pid, ct: np.ndarray, mask: Optional[np.ndarray],
                           global_bbox: Tuple[int, int, int, int],
                           out_size=(128, 128, 64), mode="train"):
    """Return cropped + resized 3D patch."""
    y_min, y_max, x_min, x_max = global_bbox

    ct_crop = normalize_ct(ct[y_min:y_max, x_min:x_max, :])
    mask_crop = mask[y_min:y_max, x_min:x_max, :] if mask is not None else np.zeros_like(ct_crop)

    factors = (out_size[0] / ct_crop.shape[0],
               out_size[1] / ct_crop.shape[1],
               out_size[2] / ct_crop.shape[2])
    ct_resize = zoom(ct_crop, factors, order=1)
    mask_resize = zoom(mask_crop, factors, order=0)
    return np.stack([ct_resize, mask_resize], axis=0)
