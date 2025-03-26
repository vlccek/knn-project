
import os


IMG_IMAGES_DIR = "../img/images"

RESPONSES_DIR = "../dataset_creating_json/responses"


outdir = "../dataset/"

def main():
    # List all files in the responses directory (annotations)
    target_paths = os.listdir(RESPONSES_DIR)

    # Generate image file paths by replacing ".json" with ".jpg"
    image_paths = os.listdir(IMG_IMAGES_DIR)

    print(image_paths)


if __name__ == "__main__":
    main()

