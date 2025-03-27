import os
import json
import shutil
import random
import subprocess
from tqdm import tqdm  # progress bar library

# Directories
IMG_IMAGES_DIR = "../img/images"
RESPONSES_DIR = "./responses"
DATASET_DIR = "./dataset"
SPLITS = {
    "train": 0.8,
    "test": 0.1,
    "validation": 0.1,
}


def main():
    split_dirs = {}
    metadata_files = {}

    # Create split directories and open metadata files
    for split in SPLITS:
        dir_path = os.path.join(DATASET_DIR, split)
        os.makedirs(dir_path, exist_ok=True)
        split_dirs[split] = dir_path
        metadata_path = os.path.join(dir_path, "metadata.jsonl")
        metadata_files[split] = open(metadata_path, "w", encoding="utf-8")

    # Get list of response files
    response_files = [f for f in os.listdir(RESPONSES_DIR) if f.endswith(".json")][:20]
    random.shuffle(response_files)
    total = len(response_files)

    train_count = int(total * SPLITS["train"])
    test_count = int(total * SPLITS["test"])
    validation_count = total - train_count - test_count

    train_files = response_files[:train_count]
    test_files = response_files[train_count:train_count + test_count]
    validation_files = response_files[train_count + test_count:]

    def process_files(file_list, split):
        # Wrap the file list in tqdm for a progress bar
        for response_file in tqdm(file_list, desc=f"Processing {split} files", unit="file"):
            base_name = os.path.splitext(response_file)[0]
            image_file_name = base_name + ".jpg"

            response_path = os.path.join(RESPONSES_DIR, response_file)
            image_src_path = os.path.join(IMG_IMAGES_DIR, image_file_name)
            image_dest_path = os.path.join(split_dirs[split], image_file_name)

            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    ground_truth_data = json.load(f)
                record = {
                    "file_name": image_file_name,
                    "ground_truth": json.dumps({"gt_parse": {"result" : ground_truth_data}})
                }
                metadata_files[split].write(json.dumps(record) + "\n")

                if os.path.exists(image_src_path):
                    shutil.copy(image_src_path, image_dest_path)
                else:
                    print(f"Warning: Image file {image_src_path} not found for {response_file}.")
            except Exception as e:
                print(f"Error processing {response_file}: {e}")

    process_files(train_files, "train")
    process_files(test_files, "test")
    process_files(validation_files, "validation")

    for f in metadata_files.values():
        f.close()

    # After processing, compress the dataset folder using pigz via tar.
    # This will create a compressed archive dataset.tar.gz
    tar_command = ["tar", "-I", "pigz", "-cf", "dataset.tar.gz", DATASET_DIR]
    try:
        subprocess.run(tar_command, check=True)
        print("Dataset folder compressed successfully using pigz into dataset.tar.gz")
    except subprocess.CalledProcessError as e:
        print("An error occurred during compression:", e)


if __name__ == "__main__":
    main()
