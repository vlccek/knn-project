import re
import os
import json
from datasets import Dataset, load_from_disk
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer, DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from datasets import load_dataset
import torch

testing = True # Creates very small dataset for testing

IMG_IMAGES_DIR = "../img/images"

RESPONSES_DIR = "../dataset_creating_json/responses"

OUTPUT_DIR_MODEL = "./my_finetuned_donut_model"

# Load the processor and model (Donut)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", use_fast=True)
# Load the model
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")


# Filter function to keep only examples where both image and annotation files exist
def filter_existing(example):
    return os.path.exists(example["image_path"]) and os.path.exists(example["target_path"])

class CustomDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding="longest"):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features):
        # Each f["pixel_values"] is now a list – convert it to a tensor
        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in features])

        # For labels, we get a list of tokenized sequences
        labels = [f["labels"] for f in features]
        # We use tokenizer.pad, which returns a tensor
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=self.padding,
            return_tensors="pt"
        )["input_ids"]

        return {"pixel_values": pixel_values, "labels": padded_labels}


def preprocess_function(example):
    # Load the image - the key "image_path" contains the path to the image file
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values[0]

    # Load the annotation - the key "target_path" contains the path to the JSON annotation file
    annotation_file = example["target_path"]
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotation_data = json.load(f)

    # Convert the annotation (JSON) to a string - adjust the format as needed for training
    target_text = json.dumps(annotation_data)

    # Tokenize the text
    labels = processor.tokenizer(
        target_text,
        add_special_tokens=True,
        truncation=True,
        padding="max_length"
    ).input_ids

    return {"pixel_values": pixel_values, "labels": labels}

def is_processed_dataset_available(path):
    return False ## tmp for testing :)

    # Required files created by save_to_disk
    required_files = ["added_tokens.json", "preprocessor_config.json", "sentencepiece.bpe.model",
                      "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    if not os.path.exists(path):
        return False
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            return False
    return True


def main():
    responses_directory = RESPONSES_DIR
    image_directory = IMG_IMAGES_DIR

    if not (is_processed_dataset_available(OUTPUT_DIR_MODEL)):
        print("Preprocessing dataset...")
        # List all files in the responses directory (annotations)
        if testing:
            target_paths = os.listdir(responses_directory)[0:10]
        else:
            target_paths = os.listdir(responses_directory)

        # Generate image file paths by replacing ".json" with ".jpg"
        image_paths = [os.path.join(image_directory, path.split(".json")[0] + ".jpg") for path in target_paths]

        # Prepend the responses directory path to annotation files
        target_paths = [os.path.join(responses_directory, path) for path in target_paths]

        data = {
            "image_path": image_paths,
            "target_path": target_paths,
        }

        dataset = Dataset.from_dict(data)

        dataset = dataset.filter(filter_existing)

        # Apply preprocessing to the entire dataset
        # dataset = dataset.map(preprocess_function, batched=True, batch_size=32)
        dataset = dataset.map(preprocess_function)

        processor.save_pretrained(OUTPUT_DIR_MODEL)

        dataset.save_to_disk(OUTPUT_DIR_MODEL)
    else:  # dataset is already preprocessed
        print("Loading preprocessed dataset...")
        dataset = load_from_disk(OUTPUT_DIR_MODEL)

    print("Preprocessing done.")

    print(f"{('-'*20)} Training the model...  {'-' * 20}")

    data_collator = CustomDataCollator(tokenizer=processor.tokenizer)

    training_args = TrainingArguments(
        output_dir="./donut-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="no",
        save_steps=1000,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR_MODEL)


if __name__ == "__main__":
    main()

