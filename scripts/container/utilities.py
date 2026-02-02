from pathlib import Path
import sys
import yaml
from PIL import Image


def load_images(images_path):
    if not images_path.exists():
        return {}

    image_dict = {}
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    for img_file in images_path.iterdir():
        if img_file.suffix.lower() not in extensions:
            continue 

        with Image.open(img_file).convert("RGB") as img:
            image_dict[img_file.stem] = img

    return image_dict


def embed_images(images, prompts):
    if not images:
        return prompts

    filled_prompts = [
        {
            "prompts": prompt_obj["prompt"],
            "multi_modal_data": {"image": images[prompt_obj["image"]]}
        }
        for prompt_obj in prompts
        ]


    return filled_prompts


def parse_prompts(prompts_path, images_path=None):
    if not prompts_path.exists():
        print(f"No prompts exist at {prompts_path}")
        print("Exitting...")
        sys.exit(1)
    with prompts_path.open("r") as f:
        prompts = yaml.safe_load(f)["prompts"]
    
    if images_path:
        prompts = embed_images(load_images(images_path), prompts)
    
    return prompts


def load_models():
    file_path = Path("/workspace/yaml/models.yaml")
    if not file_path.exists():
        print(f"No {file_path} found.")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)[f"models"]


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()