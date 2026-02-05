#!/usr/bin/env python3

"""
gemma_embeddings.py

Script for running embedding models (tested on Qwen-Embedding-4B)

"""

from vllm import LLM
import sys
import time
from runner_utilities.preprocess import load_prompts
from runner_utilities.argparse import parse_and_validate_args


def run(model, duration, iterations, prompts):
    llm = LLM(model=model, runner="pooling")

    outputs = []
    iteration_count = 0
    start = time.monotonic()

    # yield prompts cyclically until condition is no longer true
    def prompt_generator():
        def condition():
            if duration:
                return time.monotonic() - start < duration
            elif iterations:
                return iteration_count < iterations

        while True:
            for prompt in prompts:
                if condition():
                    yield prompt
                else:
                    return

    for prompt in prompt_generator():
        outputs.extend(llm.embed(prompt))
        iterations += 1

    print(
        f"Total runtime: {time.monotonic() - start:.2f}s for {iterations} iterations."
    )


def main():
    args = parse_and_validate_args(
        description="Script for running embedding models.", argv=sys.argv
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
