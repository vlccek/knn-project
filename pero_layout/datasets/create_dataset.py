import os
import json
import xml.etree.ElementTree as ET
import random

label_mapping = {
    "kapitola": 1,
    "cislo strany": 2,
    "jiny nadpis": 3,
    "jine cislo": 4,
    "podnadpis": 5,
    "nadpis v textu": 6
}

def load_files_to_skip(filename='files_to_skip.txt'):
    """Načte seznam souborů, které se mají přeskočit, ze souboru."""
    with open(filename, 'r', encoding='utf-8') as f:
        files_to_skip = [line.strip() for line in f.readlines()]
    return files_to_skip

def convert_labelstudio_bbox_to_pixels(bbox, original_width, original_height):
    """
    Convert Label Studio bounding box (normalized coordinates) to pixel coordinates.
    """
    x1 = bbox['x']/100 * original_width
    y1 = bbox['y']/100 * original_height
    x2 = x1 + bbox['width']/100 * original_width
    y2 = y1 + bbox['height']/100 * original_height
    bbox['x1'] = x1
    bbox['x2'] = x2
    bbox['y1'] = y1
    bbox['y2'] = y2

    return bbox

def calculate_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_bbox = bbox2['x1']
    x2_bbox = bbox2['x2']
    y1_bbox = bbox2['y1']
    y2_bbox = bbox2['y2']


    
    # Výpočet překrytí dvou boxů
    x_overlap = max(0, min(x2, x2_bbox) - max(x1, x1_bbox))
    y_overlap = max(0, min(y2, y2_bbox) - max(y1, y1_bbox))
    
    overlap_area = x_overlap * y_overlap
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_bbox - x1_bbox) * (y2_bbox - y1_bbox)
    
    # Vratit poměr překrytí k celkové ploše
    return overlap_area / min(area1, area2)

def get_best_matching_annotation(token_bbox, annotations):
    best_match = None
    best_overlap = 0
    
    for annotation in annotations:
        overlap = calculate_overlap(token_bbox, annotation)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = annotation
    
    return best_match


def parse_ocr_xml(ocr_file):
    """Parse the OCR XML file and extract tokens and bounding boxes."""
    tree = ET.parse(ocr_file)
    root = tree.getroot()
    
    namespaces = {'alto': 'http://www.loc.gov/standards/alto/ns-v2#'}
    
    tokens = []
    bboxes = []

    page = root.find('.//alto:Page', namespaces)
    maxheight = int(page.attrib['HEIGHT']) if page is not None else None
    maxwidth = int(page.attrib['WIDTH']) if page is not None else None

    for textblock in root.findall('.//alto:TextBlock', namespaces):
        for textline in textblock.findall('alto:TextLine', namespaces):
            for string in textline.findall('alto:String', namespaces):
                tokens.append(string.attrib['CONTENT'])
                bbox = (int(string.attrib['HPOS']), int(string.attrib['VPOS']),
                        int(string.attrib['HPOS']) + int(string.attrib['WIDTH']),
                        int(string.attrib['VPOS']) + int(string.attrib['HEIGHT']))
                bboxes.append(bbox)
    
    return tokens, bboxes, maxheight, maxwidth

def parse_labelstudio_json(labelstudio_file):
    """Parse the Label Studio JSON file and extract the annotations."""
    with open(labelstudio_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = {}
    
    for item in data:
        for annotation in item['annotations']:
            image_name = item['data']['image'].split('/')[-1]
            annotations[image_name] = []
            for res in annotation['result']:
                if res['type'] == 'relation' or len(res['value'].get('rectanglelabels', [])) == 0:
                    continue
                
                label = res['value']['rectanglelabels'][0]
                original_width = res['original_width']
                original_height = res['original_height']
                ner_tag = label_mapping.get(label, 0)
                annotations[image_name].append({
                    "bbox": res['value'],
                    "original_width" : original_width,
                    "original_height" : original_height,
                    "label": ner_tag
                })
    
    return annotations

def normalize_bboxes(bboxes, original_width, original_height):
    """
    Normalizuje seznam bounding boxů podle původní šířky a výšky obrázku.
    Pokud jsou x1 a x2 obě záporné, nastaví je na [0, 1] (nulová plocha).
    Pokud je nějaká hodnota menší než 0, nastaví ji na 0.
    """
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
        # Normalizace
        normalized_x1 = int(1000 * x1 / original_width)
        normalized_y1 = int(1000 * y1 / original_height)
        normalized_x2 = int(1000 * x2 / original_width)
        normalized_y2 = int(1000 * y2 / original_height)
        
        # Pokud jsou obě souřadnice x1 a x2 záporné, nastaví je na [0, 1]
        if normalized_x1 < 0 and normalized_x2 < 0:
            normalized_x1, normalized_x2 = 0, 1
        if normalized_y1 < 0 and normalized_y2 < 0:
            normalized_y1, normalized_y2 = 0, 1
        
        # Pokud je hodnota menší než 0, nastaví ji na 0
        normalized_bbox = [
            max(0, normalized_x1),
            max(0, normalized_y1),
            max(0, normalized_x2),
            max(0, normalized_y2)
        ]
        
        normalized_bboxes.append(normalized_bbox)
    
    return normalized_bboxes

def generate_dataset(labelstudio_json, ocr_folder, files_to_skip):
    annotations = parse_labelstudio_json(labelstudio_json)
    
    dataset = []
    i = 0
    for image_name, ann_data in annotations.items():
        # Pokud je soubor na seznamu k přeskočení, přeskočíme ho
        if image_name not in files_to_skip or len(ann_data) == 0:
            if(len(ann_data) == 0):
                print("Skipping empty annotations.")
            print(f"Soubor {image_name} se přeskočí. 🚫")
            continue

        ocr_file = os.path.join(ocr_folder, image_name.replace('.jpg', '.xml'))
        
        # Zkontrolujeme, zda XML soubor existuje
        if not os.path.exists(ocr_file):
            print(f"Soubor {ocr_file} neexistuje. 🚫")
            continue  # Pokračujeme na další obrázek, pokud soubor neexistuje
        else:
            print(f"Soubor {ocr_file} existuje. ✅")
        
        tokens, bboxes, maxheight, maxwidth = parse_ocr_xml(ocr_file)
        if ann_data[0]['original_width'] != maxwidth or ann_data[0]['original_height'] != maxheight:
            print("Skipping due to not matching max height or width")
            break
        
        # Předpokládejme, že ann_data obsahuje anotace pro všechny bounding boxy na daném obrázku
        all_ann_bboxes = []
        
        for ann in ann_data:
            ann_value = ann['bbox']
            original_width = ann['original_width']
            original_height = ann['original_height']
            
            # Přepočet na souřadnice v pixelech
            ann_bbox = convert_labelstudio_bbox_to_pixels(ann_value, original_width, original_height)
            # Přidáme do seznamu všech bounding boxů pro daný obrázek
            all_ann_bboxes.append(ann_bbox)
        
        ner_tags = []
        for token, token_bbox in zip(tokens, bboxes):
            matched_annotation = get_best_matching_annotation(token_bbox, all_ann_bboxes)
            if matched_annotation:
                ner_tags.append(label_mapping[matched_annotation['rectanglelabels'][0]])
            else:
                ner_tags.append(0)
        
        # Vytvoření záznamu v datasetu pro tento obrázek
        dataset.append({
            "tokens": tokens,
            "bboxes": normalize_bboxes(bboxes, original_width, original_height),
            "ner_tags": ner_tags,
            "image": "images/" + image_name
        })
    
    return dataset

def split_dataset(dataset, train_ratio=0.80, test_ratio=0.1, validation_ratio=0.1):
    """Randomly splits the dataset into train, test, and validation sets, using only a small fraction (20%) of the dataset."""
    
    # Zamíchání datasetu pro náhodné rozdělení
    #random.shuffle(dataset)
    
    # Celkový počet záznamů v datasetu
    total_size = len(dataset)
    
    # Určení velikosti každé sady podle zadaného poměru
    fraction_size = int(1 * total_size)  # Pouze 20% celkového datasetu
    train_size = int(train_ratio * fraction_size)
    test_size = int(test_ratio * fraction_size)
    validation_size = fraction_size - train_size - test_size

    
    # train_data = dataset[0:100]  # Prvních 5 položek
    # test_data = dataset[185:186]  # 6. až 10. položka (indexy 5 až 9)
    # validation_data = dataset[201:300] 
    # test_data = [record for record in test_data if len(record.get('tokens', [])) <= 512]
    print(len(dataset))
    # train_data = dataset[0:750]  # Prvních 5 položek
    # test_data = dataset[751:830]  # 6. až 10. položka (indexy 5 až 9)
    # validation_data = dataset[830:300] 

    train_data = dataset[0:train_size]
    test_data = dataset[train_size:train_size + test_size]
    validation_data = dataset[train_size + test_size:train_size + test_size + validation_size]
    print(len(train_data))
    print(len(test_data))
    print(len(validation_data))


    return train_data, test_data, validation_data


files_to_skip = load_files_to_skip('files_to_skip.txt')
labelstudio_json = 'annotations.json'
ocr_folder = './output/alto'

dataset = generate_dataset(labelstudio_json, ocr_folder, files_to_skip)
train_data, test_data, validation_data = split_dataset(dataset)

# Uložení každého datasetu do samostatného souboru
with open('dataset_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open('dataset_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

with open('dataset_validation.json', 'w', encoding='utf-8') as f:
    json.dump(validation_data, f, ensure_ascii=False, indent=4)