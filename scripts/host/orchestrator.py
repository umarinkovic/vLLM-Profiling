#!/usr/bin/env python3

"""
orhcestrator.py - script for automatically running all models in .config/models.yaml on all
gpus listed in gpus.yaml 
"""

import argparse
from collections import defaultdict, deque
import subprocess
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent

def parse_gpus():
    file_path = PROJECT_ROOT / ".config" / "gpus.yaml"
    if not file_path.exists():
        subprocess.run(["generate_gpu_yaml.sh"], check=True)
    with file_path.open("r") as f:
        return [gpu for gpu in yaml.safe_load(f)["gpus"] if not gpu.get("disabled", False)]
    

def parse_models():
    file_path = PROJECT_ROOT / ".config" / "models.yaml"
    if not file_path.exists():
        print("No .config/models.yaml found.")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)[f"models"]
    
def run(args):
    gpus = parse_gpus()
    models = parse_models()

    device_to_models_map = defaultdict(deque)
    for gpu in gpus:
        for model in models:
            if gpu["name"] not in model.get("disabled_on", []):
                device_to_models_map[gpu["device"]].append(model)

    device_to_name_map = {gpu["device"] : gpu["name"] for gpu in gpus}

    # TODO: implement parallelism, currently writing single threaded but will be expanded later
    # to include the possibility of starting multiple profiling tasks
    devices = list(device_to_models_map.keys())
    while any(device_to_models_map.values()):
        for device in devices:
            models_queue = device_to_models_map[device]
            if models_queue:
                model = models_queue.popleft()
                script_args = ["--model", model['name'], "--model-type", model['type']]
                subprocess.call([
                "scripts/host/docker_tool.py",
                "run", 
               # "--image-name", 
                #args.docker_image, 
                "--device",
                device,
                "--device-name",
                device_to_name_map[device],
                "--script",
                args.script, 
                "--",
                ] 
                + 
                script_args)
    


def main():
    parser = argparse.ArgumentParser("Orchestrate profiling of given models on available AMD GPUs")
    parser.add_argument("--docker-image", help = "Docker image on which to run vllm",
                        default = "hyoon11/vllm-dev:20260121_43_py3.12_torch2.9_triton3.5_navi_upstream_6a09612_ubuntu24.04")
    parser.add_argument("--num-procs", 
                        help="""One GPU always runs one profiling task at a time.
                        On a system with multiple GPUs, we can choose to run multiple 
                        profiling tasks (1 for each GPU maximum), if resource exhaustion
                        is not a problem (RAM being the main concern)""",
                        default=1)
    parser.add_argument("--script", 
                        help="Name of the script to run inside the containers.",
                        default="run_model.py")
    
    args = parser.parse_args()

    run(args)
    return


if __name__ == "__main__":
    main()