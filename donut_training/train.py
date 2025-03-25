import re
import os
import json
from datasets import Dataset
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Define directories for responses (annotations) and images
responses_directory = "../dataset_creating_json/responses"
image_directory = "../img/images"

# List all files in the responses directory (annotations)
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

# Load the processor and model (Donut)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", use_fast=True)


# Filter function to keep only examples where both image and annotation files exist
def filter_existing(example):
    return os.path.exists(example["image_path"]) and os.path.exists(example["target_path"])

dataset = dataset.filter(filter_existing)

# Preprocessing function that converts data into a format suitable for training:
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

# Apply preprocessing to the entire dataset
dataset = dataset.map(preprocess_function)

output_dir = "./my_finetuned_donut_model"

processor.save_pretrained(output_dir)

print("Preprocessing done.")


# Load the model
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

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
)

trainer.train()

model.save_pretrained(output_dir)