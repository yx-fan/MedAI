## How to prepare nnUNet data
This script prepares the nnUNet data structure from the raw data directory. It processes each case, binarizes the masks, and saves them in the appropriate format.

It is designed to handle errors gracefully, skipping cases that fail to process correctly.

## Usage
Prepare the nnUNet data by running the following command in the terminal:
```bash
./preprocess_nnunet.sh
```

Training the model can be done using the following command:
```bash
./train_nnunet_gpu.sh 2d 0 0
./train_nnunet_gpu.sh 3d_fullres 0 0
./train_nnunet_gpu.sh 3d_lowres 0 0
```
