from PeroProcessor import PeroProcessor
from LayoutProcessor import LayoutProcessor
import sys

if __name__ == "__main__":
    ocr_folder = sys.argv[1]
    image_folder = sys.argv[2]

    image_data = PeroProcessor().prepare_layout_input(ocr_folder, image_folder)
    
    preds, image = LayoutProcessor().process_image_by_layout(image_data)
    image.save("layout_processed_image.jpg")
    
    print(image_data["tokens"])
    print(preds)
    