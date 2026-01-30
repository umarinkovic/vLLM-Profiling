#!/usr/bin/env python3

"""
run_model.py 

TODO: add description, use cases etc.
"""

from vllm import LLM, SamplingParams
import torch
import os
import yaml
from pathlib import Path
import sys
import argparse
from utilities import Tee

def set_env(model, additional_env_vars=None):
    os.environ['MIOPEN_ENABLE_LOGGING'] = '1'
    os.environ['MIOPEN_ENABLE_LOGGING_CMD'] = '1'
    os.environ['HIPBLASLT_LOG_MASK'] = '32'
    os.environ['TORCH_BLAS_PREFER_HIPBLASLT'] = '1'

    # TODO: remove this env var as torch provides API for querying GPU name? (not always 100% accurate)
    device_name_safe = os.getenv("DEVICE_NAME", "GPU").replace(" ", "_")
    
    # TODO: add support for windows paths
    log_dir = Path(f'/workspace/logs/{device_name_safe}/{model.replace("/", "_")}')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "cmd.log", "w", buffering=1)
    sys.stdout = Tee(sys.stdout,log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    os.environ['HIPBLASLT_LOG_FILE'] = str(log_dir / 'hipblaslt.log')

def parse_prompts(model_type):
    prompts_path = Path(f"/workspace/prompts/{model_type}.yaml")
    if not prompts_path.exists():
        print(f"No prompts exist for model-type: {model_type}")
        print("Exitting...")
        sys.exit(1)
    with prompts_path.open("r") as f:
        return yaml.safe_load(f)["prompts"]


def main():
    parser = argparse.ArgumentParser(description="Run vLLM model inference")
    
    parser.add_argument(
        "--model",
        help="Model name",
        required=True
    )
    parser.add_argument(
        "--model-type",
        help="Model type (text, multimodal etc..)",
        required=True
    )

    parser.add_argument(
        "--no-trust",
        dest="trust",
        help="Parameter `trust_remote_code` passed to vllm (default true)",
        action="store_false",
    )
    
    # TODO: THESE SHOULD BE ENV VARIABLES READ FROM YAML 
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
        help="GPU memory fraction to use (0.0-1.0, default: 0.9)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )

    parser.add_argument(
        "--iterations",
        help="Number of inferrence iterations on the prompts to perform.",
        type=int,
        default=3
    )
    ### END OF ENV VARS
    
    args = parser.parse_args()

    prompts = parse_prompts(args.model_type)

    set_env(args.model)
    
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"GPU Memory: {args.gpu_mem_util*100}%")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")
    
    print("Loading model...")
    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=30000
    )
    print("Model loaded\n")
    
    print(f"Running {len(prompts)} test prompts {args.iterations} times")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    outputs = []
    for i in range(args.iterations):
        outputs.append(llm.generate(prompts, sampling_params))
    
    print(f"Generated {sum(len(batch) for batch in outputs)} responses\n")
    
    print(f"{'='*60}")
    print("Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()