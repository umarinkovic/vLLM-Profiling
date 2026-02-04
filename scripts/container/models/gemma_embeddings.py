#!/usr/bin/env python3

"""
gemma_embeddings.py

Script for running Google's embeddinggemmma model.

"""

from vllm import LLM
import argparse
import time
from pathlib import Path
import os
from model_utilities.preprocess import load_prompts

def run(duration, prompts):
    llm = LLM(
        model="google/embeddinggemma-300m", 
        runner="pooling",
        attention_backend="TORCH_SDPA"
    )

    outputs = []
    iterations = 0
    start = time.monotonic()

    # yield prompts cyclically until timer expires
    def prompt_generator():
        while True:
            for prompt in prompts:
                if time.monotonic() - start < duration:
                    yield prompt
                else:
                    return

    for prompt in prompt_generator():
        outputs.extend(llm.embed(prompt))
        iterations += 1

    # print(f"Sample output from {model}: {outputs[0].outputs[0].text}")
    print(f"Total runtime: {time.monotonic() - start:.2f}s for {iterations} iterations.")

def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 text models")

    parser.add_argument(
        "--duration",
        help="Duration of time (seconds) the model should be running",
        type=int,
        default=60
    )

    args, _ = parser.parse_known_args()
    prompts = load_prompts(Path("/workspace/yaml/prompts/embeddings.yaml"))
    run(args.duration, prompts)

if __name__ == "__main__":
    main()