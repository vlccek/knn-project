import os
import json
import shutil

# Directories
IMG_IMAGES_DIR = "../img/images"
RESPONSES_DIR = "./responses"
DATASET_DIR = "./dataset"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.jsonl")

def main():
    # Ensure the output dataset folder exists
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # Open the metadata file in write mode
    with open(METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        # Iterate over each JSON file in the responses folder
        for response_file in os.listdir(RESPONSES_DIR):
            if response_file.endswith(".json"):
                base_name = os.path.splitext(response_file)[0]
                image_file_name = base_name + ".jpg"
                
                response_path = os.path.join(RESPONSES_DIR, response_file)
                image_src_path = os.path.join(IMG_IMAGES_DIR, image_file_name)
                image_dest_path = os.path.join(DATASET_DIR, image_file_name)
                
                # Load the ground truth from the response JSON file
                with open(response_path, "r", encoding="utf-8") as f:
                    ground_truth_data = json.load(f)
                
                # Create the metadata record: "ground_truth" is a JSON-string with key "gt_parse"
                record = {
                    "file_name": image_file_name,
                    "ground_truth": json.dumps({"gt_parse": ground_truth_data})
                }
                # Write the record as a JSON line
                metadata_file.write(json.dumps(record) + "\n")
                
                # Copy the image to the dataset folder if it exists
                if os.path.exists(image_src_path):
                    shutil.copy(image_src_path, image_dest_path)
                else:
                    print(f"Warning: Image file {image_src_path} not found.")

if __name__ == "__main__":
    main()
