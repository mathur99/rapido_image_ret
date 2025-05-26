import os
import json
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def load_supported_extensions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return [ext.lower() for ext, enabled in data.items() if enabled]

def convert_images_in_dir_to_jpg(input_dir, output_dir, json_path):
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = load_supported_extensions(json_path)
    converted = []
    for f in os.listdir(input_dir):
        if any(f.lower().endswith(ext) for ext in supported_exts):
            input_path = os.path.join(input_dir, f)
            base_name = os.path.splitext(f)[0]
            output_path = os.path.join(output_dir, base_name + ".jpg")
            try:
                with Image.open(input_path) as img:
                    img.convert("RGB").save(output_path, "JPEG")
                converted.append(output_path)
                logger.info(f"Converted {input_path} to {output_path}")
            except Exception as e:
                logger.error(f"Failed to convert {input_path}: {e}")
    return converted
