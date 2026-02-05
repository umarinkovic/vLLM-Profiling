#!/usr/bin/env python3

"""
run_model.py

Script for running Deepseek-OCR image-to-text model.

"""

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

import os
import sys

from runner_utilities.preprocess import load_prompts, load_images, prepare_prompts
from runner_utilities.argparse import parse_and_validate_args
from runner_utilities.runner_tools import generate_and_collect


def run(model, duration, iterations, prompts):
    llm = LLM(
        model=model,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    # TODO: extract os.getenv and cast in a separate fun shared across runners
    sampling_params = SamplingParams(
        temperature=float(os.getenv("SP_TEMPERATURE")),
        max_tokens=int(os.getenv("SP_MAX_TOKENS")),
        # ngram logit processor args
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
        ),
        skip_special_tokens=False,
    )

    _ = generate_and_collect(
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
        description="Run Deepseek-OCR image-to-text model.",
        resources=True,
        argv=sys.argv,
    )

    prompts = prepare_prompts(
        load_prompts(
            args.prompts_path,
            load_images(args.resources_path),
        )
    )
    run(
        model=args.model,
        duration=args.duration,
        iterations=args.iterations,
        prompts=prompts,
    )


if __name__ == "__main__":
    main()
