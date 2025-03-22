# %%
import json
import pandas as pd
from marshal import loads
import os
from pytesseract.pytesseract import tesseract_cmd
import pytesseract

tesseract_models_folder = "../tesseract-models/"
pytesseract.pytesseract.tesseract_cmd = '/storage/brno2/home/xvlkja07/local/bin/tesseract'

# Open and load JSON file
data = {}
with open('data.json') as f:
    data = json.load(f)

# Convert JSON to pandas DataFrame
df = pd.DataFrame(data)

# Drop columns with names
df = df.drop(columns=['updated_at', "drafts", "comment_authors", "comment_count", "last_comment_updated_at",
                      "unresolved_comment_count", "total_annotations", 'total_predictions', "cancelled_annotations"])
df["data"]

# %%
# Take only first 10 rows
import concurrent.futures
import json
from PIL import Image
import pytesseract

def process_row(row):
    local_wrong_count = 0
    local_chapters = {}
    annotations = row['annotations']
    img_name = row["data"]["image"].split("images/")[-1]

    try:
        img = Image.open(f"../img/images/{img_name}")
        img.load()
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")
        return (None, {}, 0)

    if img is None:
        return (None, {}, 0)

    image_width, image_height = img.size

    # Process each annotation block
    # # TODO:
    # - [] parser diffreent data diffrently (cislo strany, cislo kapitoly, nazev urovně etc.)
    # - [] add data without relation to the relationed data
    for a in annotations:
        ocred_data = {}
        for ann in a["result"]:
            if ann.get("type") == "rectanglelabels":
                ann_id = ann.get("id")
                label = ann.get("value", {}).get("rectanglelabels", [None])
                x = ann.get("value", {}).get("x")
                y = ann.get("value", {}).get("y")
                width = ann.get("value", {}).get("width")
                height = ann.get("value", {}).get("height")

                orig_w = image_width
                orig_h = image_height
                left = int((x / 100) * orig_w)
                upper = int((y / 100) * orig_h)
                right = left + int((width / 100) * orig_w)
                lower = upper + int((height / 100) * orig_h)

                if "cislo strany" in label:
                    cropped_img = img.crop((left, upper, right, lower))
                    custom_config = f' --tessdata-dir "{tesseract_models_folder}" -l eng --oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
                    text = pytesseract.image_to_string(cropped_img, config=custom_config)

                    if text.strip() == "0" or text.strip() == "":
                        processed_img = cropped_img.convert("L")
                        processed_img = processed_img.point(lambda p: 0 if p < 128 else 255, '1')
                        retry_config = f'--tessdata-dir "{tesseract_models_folder}" -l eng+ces --oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                        ocr_text_retry = pytesseract.image_to_string(processed_img, config=retry_config).strip()
                        print("OCR failed, retrying:", ocr_text_retry)
                        text = ocr_text_retry

                    if text.strip() == "0" or text.strip() == "":
                        print(f"Image with incorrect text: {img_name}: before: \"{text}\"")
                        processed_img.save(f"../img/problematic/{img_name}-{ann_id}.jpg")
                        local_wrong_count += 1

                    ocred_data[ann_id] = {
                        "text": text.strip(),
                        "label": label,
                    }
                else:
                    cropped_img = img.crop((left, upper, right, lower))
                    custom_config = f' --tessdata-dir "{tesseract_models_folder}" -l eng+ces --oem 3 --psm 6'
                    text = pytesseract.image_to_string(cropped_img, config=custom_config)
                    ocred_data[ann.get("id")] = {
                        "text": text.strip(),
                        "label": label,
                    }

        local_chapters[img_name] = []
        for ann in a["result"]:
            if ann.get("type") == "relation":
                id1 = ann.get("from_id")
                id2 = ann.get("to_id")
                # Assume that id1 and id2 exist in ocred_data
                chapter = {
                    "name": ocred_data[id2]["text"] if id2 in ocred_data else "",
                    "page": ocred_data[id1]["text"] if id1 in ocred_data else ""
                }
                local_chapters[img_name].append(chapter)

    return (img_name, local_chapters, local_wrong_count)

# Main program run
chapters = {}
total_wrong_text = 0
cpu_count = os.cpu_count()
print(f"CPU count: {cpu_count}")

# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
    futures = [executor.submit(process_row, row) for index, row in df.iterrows()]
    for future in concurrent.futures.as_completed(futures):
        img_name, local_chapters, local_wrong_count = future.result()
        if img_name is not None:
            chapters.update(local_chapters)
            total_wrong_text += local_wrong_count

print(f"Total number of images with text errors: {total_wrong_text}")

# Save results to file
with open("chapters.json", "w") as f:
    json.dump(chapters, f)
