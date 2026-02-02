#!/usr/bin/env python3

"""
run_model.py 

Script for running Qwen3 text models.

"""

from vllm import LLM, SamplingParams
import torch
import argparse
import time
import os
import sys
from pathlib import Path
# TODO: find a better solution this is too hacky
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utilities import parse_prompts


def run(model, duration, prompts):
    llm = LLM(
        model=model,
        gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN"))
    )

    sampling_params = SamplingParams(
        temperature=float(os.getenv("SP_TEMPERATURE")),
        max_tokens=int(os.getenv("SP_MAX_TOKENS"))
    )

    outputs = []
    iterations = 0
    start = time.monotonic()

    while time.monotonic() - start < duration:
        outputs.extend(llm.generate(prompts, sampling_params))
        iterations += 1

    print(f"Sample output from {model}: {outputs[0].text}")
    print(f"Total runtime: {time.monotonic() - start:.2f}s for {iterations} iterations.")

def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 text models")

    # TODO: potentially get rid of this argument since its only used in this script
    parser.add_argument(
        "--model",
        help="Model name",
        required=True,
        type=str
    )
    parser.add_argument(
        "--duration",
        help="Duration of time (seconds) the model should be running",
        type=int,
        default=60
    )

    args, _ = parser.parse_known_args()

    if not args.model.startswith("Qwen/"):
        print("This script is only meant for running Qwen text models.")
        sys.exit(1)

    prompts = parse_prompts(Path("/workspace/yaml/prompts/text.yaml"))

    run(args.model, args.duration, prompts)

if __name__ == "__main__":
    main()