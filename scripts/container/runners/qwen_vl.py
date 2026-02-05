#!/usr/bin/env python3

"""
qwen_vl.py

Script for running Qwen3-VL models.

"""

import torch
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import argparse
import time
import os
from pathlib import Path
from model_utilities.preprocess import load_prompts, load_images, prepare_prompts


def prepare_inputs_for_vllm(prompts, processor):
    text = processor.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        prompts,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def run(duration, prompts):

    model = "Qwen/Qwen3-VL-4B-Instruct"

    llm = LLM(
        model=model,
        gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL")),
        max_model_len=int(os.getenv("MAX_MODEL_LEN")),
        mm_encoder_tp_mode="data",
        enable_expert_parallel=False,  # revisit
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=float(os.getenv("SP_TEMPERATURE")),
        max_tokens=int(os.getenv("SP_MAX_TOKENS")),
        top_k=-1,
        stop_token_ids=[],
    )

    outputs = []
    iterations = 0
    start = time.monotonic()

    while time.monotonic() - start < duration:
        outputs.extend(llm.generate(prompts, sampling_params))
        iterations += 1

    print(f"Sample output from {model}: {outputs[0].outputs[0].text}")
    print(
        f"Total runtime: {time.monotonic() - start:.2f}s for {iterations} iterations."
    )


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL models")

    parser.add_argument(
        "--duration",
        help="Duration of time (seconds) the model should be running",
        type=int,
        default=60,
    )

    args, _ = parser.parse_known_args()
    prompts = prepare_prompts(
        load_prompts(Path("/workspace/yaml/prompts/multimodal.yaml")),
        load_images(Path("/workspace/images/multimodal")),
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    parsed_prompts = [
        prepare_inputs_for_vllm(prompt, processor) for prompt in [prompts]
    ]

    run(args.duration, parsed_prompts)


if __name__ == "__main__":
    main()
