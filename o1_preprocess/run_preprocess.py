from generate_processed_data_2_d import generate_processed_data
from generate_processed_data_2_5_d import generate_processed_data as generate_processed_data_2_5d

if __name__ == '__main__':
    raw_data_dir = "data/raw/images/"
    mask_data_dir = "data/raw/masks/"
    processed_data_dir = "data/processed/"
    # generate_processed_data(raw_data_dir, mask_data_dir, processed_data_dir)
    generate_processed_data_2_5d(raw_data_dir, mask_data_dir, processed_data_dir, N=5, margin=10, out_size=(256, 256))