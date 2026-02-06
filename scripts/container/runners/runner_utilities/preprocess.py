"""

preprocess.py - utilities for pre-processing the prompts written inside the yaml folder

"""

import yaml
from PIL import Image


def load_images(images_path):
    if not images_path.exists():
        raise (FileNotFoundError(f"No images files found at {images_path}"))
    image_dict = {}
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    for img_file in images_path.iterdir():
        if img_file.suffix.lower() not in extensions:
            continue

        with Image.open(img_file).convert("RGB") as img:
            image_dict[img_file.stem] = img.resize((224, 224))

    return image_dict


def load_prompts(prompts_path):
    if not prompts_path.exists():
        raise (FileNotFoundError(f"No prompts file found at {prompts_path}"))
    with prompts_path.open("r") as f:
        return yaml.safe_load(f)["prompts"]


def prepare_prompts(prompts, multimedia_dict):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": field["type"],
                    field["type"]: (
                        field["text"]
                        if field["type"] == "text"
                        else multimedia_dict[field["name"]]
                    ),
                }
                for field in content
            ],
        }
        for content in prompts
    ]
