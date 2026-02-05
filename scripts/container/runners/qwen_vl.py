#!/usr/bin/env python3

"""
qwen_vl.py

Script for running Qwen3-VL models.

"""

import torch
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import sys
import time
import os
from runner_utilities.preprocess import load_prompts, prepare_prompts, load_images
from runner_utilities.argparse import parse_and_validate_args
from runner_utilities.runner_tools import generate_and_collect


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


def run(model, duration, iterations, prompts):

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
        description="Script for running Qwen-VL models.", resources=True, argv=sys.argv
    )

    prompts = prepare_prompts(
        load_prompts(args.prompts_path),
        load_images(args.media_path),
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    parsed_prompts = [
        prepare_inputs_for_vllm(prompt, processor) for prompt in [prompts]
    ]

    run(
        model=args.model,
        duration=args.duration,
        iterations=args.iterations,
        prompts=parsed_prompts,
    )


if __name__ == "__main__":
    main()
