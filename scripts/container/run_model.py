#!/usr/bin/env python3

"""
run_model.py

Script for setting up the environment and running of models via script passed through positional cmd-line argument

"""

import os
from pathlib import Path
import subprocess
import sys
import argparse
from utilities import Tee
import torch


GPU_NAME = ""
LOG_FILE = ""


def setup_environment(model, **custom_env_vars):
    # TODO: add support for windows paths
    global GPU_NAME
    GPU_NAME = os.getenv("DEVICE_NAME", "GPU").replace(" ", "_")
    log_dir = Path(f'/workspace/logs/{GPU_NAME}/{model.replace("/", "_")}')
    log_dir.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = open(log_dir / "cmd.log", "w", buffering=1)
    sys.stdout = Tee(sys.stdout, LOG_FILE)
    sys.stderr = Tee(sys.stderr, LOG_FILE)

    env = {
        "MIOPEN_ENABLE_LOGGING": "1",
        "MIOPEN_ENABLE_LOGGING_CMD": "1",
        "HIPBLASLT_LOG_MASK": "32",
        "TORCH_BLAS_PREFER_HIPBLASLT": "1",
        "HIPBLASLT_LOG_FILE": str(log_dir / "hipblaslt.log"),
        # default config
        "GPU_MEM_UTIL": os.getenv("GPU_MEM_UTIL", "0.85"),
        "SP_TEMPERATURE": os.getenv("SP_TEMPERATURE", "0.5"),
        "SP_MAX_TOKENS": os.getenv("SP_MAX_TOKENS", "1024"),
    }

    env.update(custom_env_vars)

    os.environ.update(env)
    # TODO: dont return env?
    return env


def run(model, script, extra_args):
    global GPU_NAME
    environment = setup_environment(model)

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"GPU Memory: {float(environment['GPU_MEM_UTIL']) * 100}%")
    print(f"Max Tokens: {int(environment['SP_MAX_TOKENS'])}")
    print(f"Temperature: {float(environment['SP_TEMPERATURE'])}")
    print(f"{'='*60}\n")

    print(f"GPU: {GPU_NAME}")
    print(f"PyTorch: {torch.__version__}\n")

    print(f"Calling: {script}")
    subprocess.run(
        [f"/workspace/scripts/runners/{script}", "--model", model, *extra_args],
        check=True,
        stdout=LOG_FILE.fileno(),
        stderr=LOG_FILE.fileno(),
        pass_fds=(LOG_FILE.fileno(),),
    )

    print(f"{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    return


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
