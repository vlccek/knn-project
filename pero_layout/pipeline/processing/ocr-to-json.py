import json

from dill import settings
from numpy.ma.extras import average

from PeroProcessor import PeroProcessor
from LayoutProcessor import LayoutProcessor
import sys
import math
import numpy as np
import os
from tqdm import tqdm


class Ocr2JsonSettings:
    distance_between_words = 0.1


def is_on_same_line(bbox1, bbox2, min_distance=Ocr2JsonSettings.distance_between_words):
    """
    Check if two bounding boxes are on the same line based on their y-coordinates.
    """
    y1 = bbox1[1]
    y2 = bbox2[1]
    height1 = bbox1[3] - bbox1[1]
    height2 = bbox2[3] - bbox2[1]
    value = abs(y1 - y2) < min_distance * max(height1, height2)

    return abs(y1 - y2) < min_distance * max(height1, height2)


def merge_title_words(data, title_classification='jiny nadpis'):
    related_words = [(class_name, bbox, token) for class_name, bbox, token in data if
                     class_name == title_classification]

    merged = []
    used_indices = set()

    for i, (A_class_name, A_bbox, A_token) in enumerate(related_words):
        if i in used_indices:
            continue

        new_token = A_token
        new_bbox = A_bbox[:]
        used_indices.add(i)

        for j, (B_class_name, B_bbox, B_token) in enumerate(related_words):
            if j == i or j in used_indices:
                continue

            if is_on_same_line(A_bbox, B_bbox):
                new_token += " " + B_token
                new_bbox = [
                    min(new_bbox[0], B_bbox[0]),
                    min(new_bbox[1], B_bbox[1]),
                    max(new_bbox[2], B_bbox[2]),
                    max(new_bbox[3], B_bbox[3]),
                ]
                used_indices.add(j)

        merged.append((A_class_name, new_bbox, new_token))

    return merged


def avg_line_height(data):
    heights = []
    for i in data:
        heights.append(i[1][3] - i[1][1])

    return sum(heights) / len(heights) if heights else 0


def find_nearest(src, list_of_targets):
    def vertical_distance_function(obj1):
        bbox1 = obj1[1]  # [x1a, y1a, x2a, y2a]
        bbox2 = src[1]  # [x1b, y1b, x2b, y2b]

        center_y1 = (bbox1[1] + bbox1[3]) / 2
        center_y2 = (bbox2[1] + bbox2[3]) / 2

        # Výpočet eukleidovské vzdálenosti mezi středy
        vertical_distance = abs(center_y1 - center_y2)
        if vertical_distance == 0:
            return 1e-10
        return vertical_distance

    minimum = min(list_of_targets, key=vertical_distance_function)

    # return the minimum element index
    min_index = list_of_targets.index(minimum)

    return min_index


def distance_function(bbox1, bbox2):
    # Calculate the distance between the centers of two bounding boxes
    center_y1 = (bbox1[1] + bbox1[3]) / 2
    center_y2 = (bbox2[1] + bbox2[3]) / 2

    center_x1 = (bbox1[0] + bbox1[2]) / 2
    center_x2 = (bbox2[0] + bbox2[2]) / 2

    distance = math.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance


def find_element_on_same_line(src, list_of_targets):
    bbox = src[1]
    y_cords = (bbox[1] + bbox[3]) / 2

    output_candidates = []

    for i, (A_class_name, A_bbox, A_token) in enumerate(list_of_targets):
        y_range = A_bbox[1], A_bbox[3]

        if y_range[0] <= y_cords <= y_range[1]:
            output_candidates.append((A_class_name, A_bbox, A_token))

    def distanec_fix_param(obj1):
        "distance btween obj1 and bbox"
        bbox1 = obj1[1]
        bbox2 = bbox

        return distance_function(bbox1, bbox2)

    if len(output_candidates) == 0:
        return None
    return min(output_candidates, key=distanec_fix_param)


def match_heading_page_chapter(headings, pages, chapters):
    """
    Match headings with their corresponding pages and chapters.
    """
    output = []

    for i, obj in enumerate(headings):
        page = find_element_on_same_line(obj, pages)
        chapter = find_element_on_same_line(obj, chapters)

        output.append((obj, page, chapter))

    return output


def map_repr_json(obj):
    """
    Map the object to a JSON representation.
    """
    title = obj[0][2]
    page = obj[1][2] if obj[1] else ""
    chapter = obj[2][2] if obj[2] else ""

    return {
        "title": title,
        "chapter_number": chapter,
        "page_number": page,
        "children": []
    }


if __name__ == "__main__":
    ocr_folder = sys.argv[1]
    image_folder = sys.argv[2]
    output_folder = "./output/structure/"
    print("Starting Model output to JSON conversion...")
    print(f"The output folder is: {os.path.abspath(output_folder)}")

    for image_file in tqdm(os.listdir(image_folder), desc="Zpracování obrázků"):
        print("Processing " + str(image_file))
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, image_file)

            image_data = PeroProcessor().prepare_layout_input(ocr_folder, image_path)

            # batch processing of long sequences
            if len(image_data["tokens"]) > 200:
                max_tokens = 200
                all_preds = []

                num_batches = (len(image_data["tokens"]) + max_tokens - 1) // max_tokens

                for i in range(num_batches):
                    start_idx = i * max_tokens
                    end_idx = min((i + 1) * max_tokens, len(image_data["tokens"]))

                    batch_image_data = {
                        "tokens": image_data["tokens"][start_idx:end_idx],
                        "bboxes": image_data["bboxes"][start_idx:end_idx],
                        "image_path": image_path
                    }

                    preds, image = LayoutProcessor().process_image_by_layout(batch_image_data)
                    all_preds.extend(preds)

                    # image.save(f"output/processed/layout_processed_image_{image_file}_{i}.jpg")

                preds = all_preds
                print(f"Prediction for {image_file} (total): {all_preds}")
            else:
                preds, image = LayoutProcessor().process_image_by_layout(image_data)
                # image.save(f"layout_processed_image_{image_file}.jpg")

            print(f"Prediction for {image_file}: {preds}")

        # image.save("layout_processed_image.jpg")
        tokens = image_data["tokens"]
        bboxes = image_data["bboxes"]

        cleaned = [(class_name, bbox, token) for class_name, bbox, token in zip(preds, bboxes, tokens) if
                   class_name != "trash"]

        avg_line_height2 = avg_line_height(cleaned)  # Average line height,

        headings = merge_title_words(cleaned, "kapitola")
        heading_scd = merge_title_words(cleaned, "jiny nadpis")
        page_number = merge_title_words(cleaned, "cislo strany")
        chapter_number = merge_title_words(cleaned, "jine cislo")

        heading_main = match_heading_page_chapter(headings, page_number, chapter_number)

        json_data = []

        for i, obj in enumerate(heading_main):
            json_data.append(map_repr_json(obj))

        heading_scd = match_heading_page_chapter(heading_scd, page_number, chapter_number)

        for i in heading_scd:
            h, page, chapter = i

            parent_index = find_nearest(h, headings)

            json_data[parent_index]["children"].append(map_repr_json(i))

        # Save the JSON data to a file
        output_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.json")
        os.makedirs(output_folder, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)


        # print(json.dumps(json_data))
