import os
from PIL import Image

def convert_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = Image.open(image_path)

                if image.mode == 'L':
                    print(f"Converting {filename} to RGB")
                    image = image.convert('RGB')

                output_path = os.path.join(output_folder, filename)
                image.save(output_path)

            except Exception as e:
                print(f"Error while processing image {filename}: {e}")

input_folder = './images'
output_folder = './images_converted'

convert_images(input_folder, output_folder)
