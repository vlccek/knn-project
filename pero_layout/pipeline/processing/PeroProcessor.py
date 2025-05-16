import os
import xml.etree.ElementTree as ET

class PeroProcessor:
    def _parse_ocr_xml(self, ocr_file):
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

    def _normalize_bboxes(self, bboxes, original_width, original_height):
        normalized_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            normalized_x1 = int(1000 * x1 / original_width)
            normalized_y1 = int(1000 * y1 / original_height)
            normalized_x2 = int(1000 * x2 / original_width)
            normalized_y2 = int(1000 * y2 / original_height)
            
            if normalized_x1 < 0 and normalized_x2 < 0:
                normalized_x1, normalized_x2 = 0, 1
            if normalized_y1 < 0 and normalized_y2 < 0:
                normalized_y1, normalized_y2 = 0, 1
            
            normalized_bbox = [
                max(0, normalized_x1),
                max(0, normalized_y1),
                max(0, normalized_x2),
                max(0, normalized_y2)
            ]
            
            normalized_bboxes.append(normalized_bbox)
        
        return normalized_bboxes

    def prepare_layout_input(self, ocr_folder, image_path):
        image_filename = os.path.basename(image_path)
        
        ocr_filename = os.path.splitext(image_filename)[0] + ".xml"
        
        ocr_file = os.path.join(ocr_folder, ocr_filename)
        
        tokens, bboxes, maxheight, maxwidth = self._parse_ocr_xml(ocr_file)
        
        bboxes = self._normalize_bboxes(bboxes, maxwidth, maxheight)

        image_data = {
            "bboxes": bboxes,
            "tokens": tokens,
            "image_path": image_path
        }

        return image_data
