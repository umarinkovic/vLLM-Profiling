#!/usr/bin/env python3

"""
qwwen_text.py

Script for running Qwen3 text models.

"""

from vllm import LLM, SamplingParams
import torch

import os
import sys
from pathlib import Path
from runner_utilities.preprocess import load_prompts
from runner_utilities.argparse import parse_and_validate_args
from runner_utilities.runner_tools import generate_and_collect


def run(model, duration, iterations, prompts):
    llm = LLM(
        model=model,
        gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN")),
    )

    sampling_params = SamplingParams(
        temperature=float(os.getenv("SP_TEMPERATURE")),
        max_tokens=int(os.getenv("SP_MAX_TOKENS")),
    )

    outputs = generate_and_collect(
        model=model,
        duration=duration,
        iterations=iterations,
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        print_example=True,
    )


def main():
    args = parse_and_validate_args(
        description="Script for running Qwen3 textual models.",
        argv=sys.argv,
    )

    prompts = load_prompts(args.prompts_path)
    run(
        model=args.model,
        duration=args.duration,
        iterations=args.iterations,
        prompts=prompts,
    )


if __name__ == "__main__":
    main()
