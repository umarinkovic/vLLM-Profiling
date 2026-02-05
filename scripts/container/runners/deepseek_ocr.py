#!/usr/bin/env python3

"""
run_model.py 

Script for running Deepseek-OCR image-to-text model.

"""

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
import argparse
import time
from pathlib import Path
import os

from model_utilities.preprocess import load_prompts, load_images, prepare_prompts


def run(duration, prompts):
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )   

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
    
    outputs = []
    iterations = 0
    start = time.monotonic()

    while time.monotonic() - start < duration:
        outputs.extend(llm.generate(prompts, sampling_params))
        iterations += 1

    print(f"Sample output from Deepseek-OCR: {outputs[0].outputs[0].text}")
    print(f"Total runtime: {time.monotonic() - start:.2f}s for {iterations} iterations.")


def main():
    parser = argparse.ArgumentParser(description="Run Deepseek-OCR image-to-text model.")
    
    parser.add_argument(
        "--duration",
        help="Duration of time (seconds) the model should be running",
        type=int,
        default=60
    )
    
    args, _ = parser.parse_known_args()

    prompts = prepare_prompts(load_prompts((Path("/workspace/yaml/prompts/image-to-text.yaml")), load_images(Path("/workspace/images/image-to-text"))))
    run(args.duration, prompts)

if __name__ == "__main__":
    main()