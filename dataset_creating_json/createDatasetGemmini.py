import os
import base64
import json
from PIL import Image  # Pro kontrolu, zda je soubor obrázek

from google import genai


# Konfigurace (stejná jako dříve)
IMAGE_DIRECTORY = "../img/images"
API_KEY = ""
MODEL = "gemini-2.0-flash"
PROMPT = """
Analyze scanned book page image, extract hierarchical chapter structure to JSON. Correct OCR errors, identify titles, chapter numbers, and starting page numbers.

OCR & Document Analysis Task

Extract chapter structure from a scanned book page image into a JSON format.

Input

Scanned book page image (potentially skewed, with varying lighting, and OCR errors).

Output

A JSON representation of the book's chapter structure (hierarchical). Each chapter/section/subsection object should contain:

title: (String) Chapter/section/subsection title (corrected for OCR errors).

chapter_number: (String, optional) Chapter/section/subsection number (e.g., "Chapter 3", "Section 2.1"). Empty string if absent.

page_number: (Integer) Starting page number.

Instructions

OCR Correction: Mitigate OCR errors using spell-checking, context, and common mistake recognition.

Chapter/Section Identification: Use font size/style, placement, keywords ("Chapter", "Section", etc. in different languages), and numbering patterns to identify headings.

Hierarchical Structure: Determine chapter/section hierarchy based on indentation, numbering, and keywords.

Page Number Extraction: Extract page numbers from various locations/formats. Extact the numer as STRING, in some cases the page could be range.

Error Handling: Use placeholders ("Untitled Chapter") for unextractable titles. Set page_number to empty string if undetectable. Prioritize accuracy over completeness.

Output Format: Valid JSON string. Without ANY syntatic sugar (e.g., comments, trailing commas, md , etc.).

Please format the chapter to hiearchical structure something like
[
{name: "Chapter 1", page: "1", children: [
{name: "Section 1.1", page: "2-5", children: []}
]

Return [] if no chapters are identified.
"""

OUTPUT_DIRECTORY = "responses"


def is_image(file):
    """Checks if the given file is an image."""
    try:
        Image.open(file)
        return True
    except:
        return False


def send_to_ai_studio(image_base64, prompt):
    """Sends an image and prompt to Google AI Studio using the genai library."""

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL)

    image_data = base64.b64decode(image_base64)
    image_part = {"mime_type": "image/jpeg", "data": image_data}  # Assuming JPEG

    contents = [prompt, image_part]
    try:
        response = model.generate_content(contents)  # Removed model parameter
        response.resolve()  # Raises exception if there is a problem with the result
        return response.text
    except Exception as e:
        print(f"Error from Gemini API: {e}")
        return None


def process_image(image_path, prompt, output_directory):
    """Processes a single image: loads, encodes, sends to AI Studio, and saves the response."""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            if os.path.basename(image_path).split(".jpg")[0] + ".json" in os.listdir(output_directory):
                print(f"Response for {image_path} already exists.")
                return

        response = send_to_ai_studio(image_base64, prompt)

        if response:
            # Saving the response to a JSON file
            file_name = os.path.basename(image_path).split(".jpg")[0] + ".json"

            file_path = os.path.join(output_directory, file_name)

            with open(file_path, "w", encoding="utf-8") as file:
                # Save the response to JSON
                file.write(response[7:-3])
            print(f"Response for {image_path} saved to {file_path}")

        else:
            print(f"Processing image {image_path} failed.")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def main():
    """Main function for iterating through images and processing them."""

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    for file_name in os.listdir(IMAGE_DIRECTORY):
        file_path = os.path.join(IMAGE_DIRECTORY, file_name)

        if os.path.isfile(file_path) and is_image(file_path):
            print(f"Processing {file_name}...")
            process_image(file_path, PROMPT, OUTPUT_DIRECTORY)
        else:
            print(f"Skipping {file_name} (not a file or not an image).")

if __name__ == "__main__":
    # Example usage
    import google.generativeai as genai

    if not os.path.exists(IMAGE_DIRECTORY):
        print("Please create the 'images' directory and put your images in it.")
    else:
        main()