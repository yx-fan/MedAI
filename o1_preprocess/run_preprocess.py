from generate_processed_data import generate_processed_data

if __name__ == '__main__':
    raw_data_dir = "data/raw/images/"
    mask_data_dir = "data/raw/masks/"
    processed_data_dir = "data/processed/"
    generate_processed_data(raw_data_dir, mask_data_dir, processed_data_dir)