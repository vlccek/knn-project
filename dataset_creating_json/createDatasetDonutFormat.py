import os
import json
import shutil
import random
import subprocess
from tqdm import tqdm  # progress bar library

# Directories
IMG_IMAGES_DIR = "./dataset_obsahy/images/"
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
    response_files = [f for f in os.listdir(RESPONSES_DIR) if f.endswith(".json")]
    print(f"Found {len(response_files)} response files in {RESPONSES_DIR}")
    random.shuffle(response_files)
    total = len(response_files)

    not_gt_array = []

    train_count = int(total * SPLITS["train"])
    test_count = int(total * SPLITS["test"])
    validation_count = total - train_count - test_count

    train_files = response_files[:train_count]
    test_files = response_files[train_count:train_count + test_count]
    validation_files = response_files[train_count + test_count:]
    print(f"Number of files: {len(train_files)} train, {len(test_files)} test, {len(validation_files)} validation")

    def process_files(file_list, split):
        # Wrap the file list in tqdm for a progress bar
        for response_file in tqdm(file_list, desc=f"Processing {split} files", unit="file"):
            # print(os.listdir(IMG_IMAGES_DIR))

            if response_file.replace(".json", ".jpg") not in os.listdir(IMG_IMAGES_DIR):
                # print(f"Image file not found for {response_file}. Skipping.")
                continue

            base_name = os.path.splitext(response_file)[0]
            image_file_name = base_name + ".jpg"

            response_path = os.path.join(RESPONSES_DIR, response_file)
            image_src_path = os.path.join(IMG_IMAGES_DIR, image_file_name)
            image_dest_path = os.path.join(split_dirs[split], image_file_name)

            try:
                with open(response_path, "r", encoding="utf-8") as f:
                    ground_truth_data = json.load(f)

                if not ground_truth_data:
                    not_gt_array.append(image_file_name)
                    print(f"No ground truth data for {image_file_name}")
                    continue
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
    tar_command = ["tar", "-I", "pigz", "-cf", "dataset_donut.tar.gz", DATASET_DIR]
    try:
        subprocess.run(tar_command, check=True)
        print("Dataset folder compressed successfully using pigz into dataset.tar.gz")
    except subprocess.CalledProcessError as e:
        print("An error occurred during compression:", e)

    print(f"Not GT array: {not_gt_array}")


if __name__ == "__main__":
    main()
