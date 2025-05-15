from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast, LayoutLMv3Processor
import torch

class LayoutProcessor:

    def _denormalize_bboxes(self, bbox, original_width, original_height):
        return [
            original_width * (bbox[0] / 1000),
            original_height * (bbox[1] / 1000),
            original_width * (bbox[2] / 1000),
            original_height * (bbox[3] / 1000),
        ]

    def process_image_by_layout(self, image_data):
        labels = ["trash", "kapitola", "cislo strany", "jiny nadpis", "jine cislo", "podnadpis", "nadpis v textu"]
        id2label = {v: k for v, k in enumerate(labels)}

        tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained("layoutlmv3")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        image = Image.open(image_data["image_path"])
        image = image.convert("RGB")

        words = image_data["tokens"]
        boxes = image_data["bboxes"]
        encoding = processor(image, words, boxes=boxes, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        offset_mapping = encoding.pop('offset_mapping')

        for k, v in encoding.items():
            encoding[k] = v.to(device)

        outputs = model(**encoding)

        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        token_boxes = encoding.bbox.squeeze().tolist()

        inp_ids = encoding.input_ids.squeeze().tolist()
        inp_words = [tokenizer.decode(i) for i in inp_ids]

        width, height = image.size
        is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0

        true_predictions = [id2label[pred]
                            for idx, pred in enumerate(predictions) if not is_subword[idx]]

        true_boxes = [self._denormalize_bboxes(box, width, height) for idx, box in enumerate(
            token_boxes) if not is_subword[idx]]

        true_words = []
        for id, i in enumerate(inp_words):
            if not is_subword[id]:
                true_words.append(i)
            else:
                true_words[-1] = true_words[-1]+i

        true_predictions = true_predictions[1:-1]
        true_boxes = true_boxes[1:-1]
        true_words = true_words[1:-1]

        preds = []
        l_words = []
        bboxes = []

        for i, j in enumerate(true_predictions):
            if true_boxes[i] not in bboxes:
                preds.append(true_predictions[i])
                l_words.append(true_words[i])
                bboxes.append(true_boxes[i])

        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()

        label2color = {"trash": 'red', "kapitola": 'green',
                    "cislo strany": 'blue', "jiny nadpis": 'orange', 'jine cislo': "purple"}
        
        for prediction, box in zip(preds, bboxes):
            draw.rectangle(box, outline=label2color[prediction], fill=(
                255, 255, 0, int(0.4 * 255)))
            draw.text((box[0]+10, box[1]-10), text=prediction,
                    fill=label2color[prediction], font=font)

        return preds, image
