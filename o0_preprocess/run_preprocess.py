import argparse
from generate_dataset import generate_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CT/Mask data for segmentation & feature extraction")

    parser.add_argument("--images_dir", type=str, required=True, help="Path to CT images (.nii.gz)")
    parser.add_argument("--masks_dir", type=str, default=None, help="Path to masks (.nii.gz), required if mode=train")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True,
                        help="'train' needs masks, 'predict' does not")
    parser.add_argument("--format", type=str, choices=["2d", "2.5d", "3d"], default="2.5d",
                        help="Data format for processing")
    parser.add_argument("--N", type=int, default=5, help="Number of stacked slices for 2.5D (odd number)")
    parser.add_argument("--margin", type=int, default=10, help="Margin (pixels) around tumor ROI")
    parser.add_argument("--out_size", type=int, nargs=2, default=(256, 256), help="Output size (H, W) for 2D/2.5D")
    parser.add_argument("--out_size_3d", type=int, nargs=3, default=(128, 128, 64),
                        help="Output size (H, W, D) for 3D patches")
    parser.add_argument("--background_ratio", type=float, default=0.1,
                        help="Proportion of background slices to keep in training mode")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation (rotation, flip, brightness change) in training mode")

    args = parser.parse_args()

    generate_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        format=args.format,
        N=args.N,
        margin=args.margin,
        out_size=tuple(args.out_size),
        out_size_3d=tuple(args.out_size_3d),
        background_ratio=args.background_ratio,
        augment=args.augment
    )
