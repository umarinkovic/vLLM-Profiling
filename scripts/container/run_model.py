#!/usr/bin/env python3

"""
run_model.py

Container script for model environment setup.

"""

import os
from pathlib import Path
import subprocess
import sys
import argparse
import traceback
from collections import Counter
from utilities import Tee


# TODO: move this to docker_tool.py; re-asses whether this script is needed or if commonalities can be
# mut into a seperate file, env setup in dockertool, and runners executed directly
def setup_environment(model, **custom_env_vars):
    # TODO: add support for windows paths

    gpu_name = os.getenv("DEVICE_NAME", "GPU").replace(" ", "_")
    log_dir = Path(f'/workspace/logs/{gpu_name}/{model.replace("/", "_")}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "cmd.log", "w", buffering=1)
    hipblaslt_log_path = str(log_dir / "hipblaslt.log")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    env = {
        "MIOPEN_ENABLE_LOGGING": "1",
        "MIOPEN_ENABLE_LOGGING_CMD": "1",
        "HIPBLASLT_LOG_MASK": "32",
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
        "HIPBLASLT_LOG_FILE": hipblaslt_log_path,
        # default config
        "GPU_MEM_UTIL": os.getenv("GPU_MEM_UTIL", "0.85"),
        "SP_TEMPERATURE": os.getenv("SP_TEMPERATURE", "0.5"),
        "SP_MAX_TOKENS": os.getenv("SP_MAX_TOKENS", "1024"),
    }

    env.update(custom_env_vars)

    os.environ.update(env)

    return (log_file, hipblaslt_log_path, gpu_name)


def sort_hipblaslt_log(hipblaslt_log_path):
    log_path = Path(hipblaslt_log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"{log_path} does not exist")

    sorted_log_path = log_path.parent / "sorted_hipblaslt.log"

    line_counter = Counter()
    with log_path.open("r") as f:
        for line in f:
            line_counter[line] += 1

    # descending sort by frequency
    sorted_lines = line_counter.most_common()

    with sorted_log_path.open("w") as f:
        for line, count in sorted_lines:
            f.write(f"{count} {line}")


def run(model, script, extra_args):
    log_file, hipblaslt_log_path, gpu_name = setup_environment(model)
    import torch

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    environment = {
        key: value for key, value in os.environ.copy().items() if key != "HF_TOKEN"
    }
    # TODO: fix this, there shouldn't be a hard-check for HF_TOKEN but env vars
    # we don't want to expose via logs (like HF_TOKEN and other secrets) should be read
    # from somewhere, like tokens.yaml
    # although I'm not a fan of parsing config yamls once inside the container
    # possible solution is to set everything in docker_tool including the hidden env_vars
    # and get rid of setup_environment here altogether
    print(f"ENVIRONMENT: {environment}")
    print(f"{'='*60}\n")

    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}\n")

    print(f"Calling: {script}")
    proc = subprocess.Popen(
        [f"/workspace/scripts/runners/{script}", "--model", model, *extra_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )

    for line in proc.stdout:
        sys.stdout.write(line)

    result = proc.wait()
    proc.stdout.close()

    if result != 0:
        print(f"{'='*60}")
        print(f"Subprocess failed: returncode={proc.returncode}, args={proc.args}")
        traceback.print_stack()
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")
        print("Complete!")
        print(f"{'='*60}")

    # TODO: fix this, remove Tee-ing from setup environment and use with f open(...)
    sys.stdout.flush()
    sys.stderr.flush()
    log_file.close()
    sort_hipblaslt_log(hipblaslt_log_path=hipblaslt_log_path)


def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference on a model.")
    parser.add_argument("--model", help="Model name", required=True, type=str)
    parser.add_argument(
        "--script", help="Script for running the model.", required=True, type=str
    )

    args, extra_args = parser.parse_known_args()

    run(args.model, args.script, extra_args)


if __name__ == "__main__":
    main()
