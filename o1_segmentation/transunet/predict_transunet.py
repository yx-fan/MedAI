import os, torch, nibabel as nib, numpy as np, torch.nn.functional as F
from transunet_config import *
from transunet_model import TransUNet

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

model = TransUNet(img_size=IMG_SIZE, patch_size=VIT_PATCH_SIZE,
                  emb_size=VIT_EMBED_DIM, depth=VIT_DEPTH,
                  heads=VIT_HEADS, num_classes=NUM_CLASSES).to(device)
ckpt = os.path.join(MODEL_SAVE_DIR, "transunet_best.pth")  # <-- match trainer
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

os.makedirs("./predictions_transunet", exist_ok=True)

for fname in os.listdir(IMAGES_TS):
    img_nii = nib.load(os.path.join(IMAGES_TS, fname))
    img = img_nii.get_fdata().astype(np.float32)       # [H,W,D]
    H, W, D = img.shape
    # normalize per-volume
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    pred_vol = np.zeros((H, W, D), dtype=np.uint8)
    for z in range(D):
        sl = img[..., z]                               # [H,W]
        sl_t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
        # resize to model size
        sl_t = F.interpolate(sl_t, size=IMG_SIZE, mode="bilinear", align_corners=False)
        with torch.no_grad():
            logits = model(sl_t)                       # [1,C,h,w] == [1,C,128,128]
            pred   = torch.argmax(logits, dim=1).float()  # [1,128,128]
            # resize back to original slice size
            pred   = F.interpolate(pred.unsqueeze(1), size=(H, W), mode="nearest").squeeze().cpu().numpy().astype(np.uint8)
        pred_vol[..., z] = pred

    pred_nii = nib.Nifti1Image(pred_vol, affine=img_nii.affine, header=img_nii.header)
    nib.save(pred_nii, os.path.join("./predictions_transunet", fname))
