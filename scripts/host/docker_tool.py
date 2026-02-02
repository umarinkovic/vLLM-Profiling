#!/usr/bin/env python3

# docker_tool.py is a script used to build and run docker images 

import argparse
import subprocess
import sys
from pathlib import Path
import os
import yaml

def prepare_env():
    path_to_env_vars = Path(__file__).parent.parent.parent / "yaml" / "env_vars.yaml"
    with open(path_to_env_vars, "r") as f:
        env_vars_to_propagate = yaml.safe_load(f)["env_vars"]
    docker_env = {env_var: os.getenv(env_var) for env_var in env_vars_to_propagate if os.getenv(env_var)}
    return docker_env

def run_container(args, script_args):
    env = prepare_env()
    cmd = [
            "docker",
            "run",
            # first, unpack environment key-value pairs into a list of k=v strings
            # then unpack the kv pairs into lists of ["-e", kv]
            *[
                item
                for key, value in env.items()
                for item in ("-e", f"{key}={value}")
            ],
            "--rm",
            "--ipc=host",
            "--group-add", "video",
            "--cap-add=SYS_PTRACE",
            "--security-opt", "seccomp=unconfined",
            "--device", "/dev/kfd",
            "-w", "/workspace"
            ]
    
    interactive = not args.script
    if interactive:
        cmd.append("-it")
    """TODO: remove or fix this, needed for parallelism probably
    else:
        cmd.append("-d") """

    # mount appropriate gpu
    cmd.extend(["--device", args.device])

    # set env var to remember device name
    cmd.extend(["-e", f'DEVICE_NAME={args.device_name}'])

    # mount dirs from host to container
    container_workspace = Path("/workspace")

    # huggingface cache dir
    hf_cache_container = "/root/.cache/huggingface"   
    Path(args.hf_cache_dir).mkdir(parents=True, exist_ok=True) 
    cmd.extend(["-v", f"{args.hf_cache_dir}:{hf_cache_container}"])
    print(f"Mounting HuggingFace cache: {args.hf_cache_dir} -> {hf_cache_container}")

    # local container scripts dir
    scripts_container = str(container_workspace / "scripts")
    cmd.extend(["-v", f"./scripts/container:{scripts_container}"])
    print(f"Mounting ./scripts/container -> {scripts_container}")

    # local prompts dir
    prompts_container = str(container_workspace / "prompts")
    cmd.extend(["-v", f"./prompts:{prompts_container}"])
    print(f"Mounting ./prompts -> {prompts_container}")

    # logs dir
    # TODO: parametrize this / add env var for customization?
    logs_container = str(container_workspace / "logs")
    cmd.extend(["-v", f"./.logs:{logs_container}"])
    print(f"Mounting ./.logs -> {logs_container}")

    # images dir
    images_container = str(container_workspace / "images")
    cmd.extend(["-v", f"./images:{images_container}"])
    print(f"Mounting ./images -> {images_container}")

    # yaml dir
    yaml_container = str(container_workspace / "yaml")
    cmd.extend(["-v", f"./yaml:{yaml_container}"])
    print(f"Mounting ./yaml -> {yaml_container}")

    shell_cmd = []

    shell_cmd.append(
        f"pip install --no-cache-dir -r {scripts_container}/requirements.txt"
    )

    if args.script:
        # TODO: enable executing any script not just python (parsing currently gets messed up so we use python3 explicitly)
        # full_script_cmd = [f"/workspace/scripts/{args.script}"] + script_args[1:]
        # full_script_str = " ".join(shlex.quote(arg) for arg in full_script_cmd)
        # cmd.extend(["/bin/bash", "-c", full_script_str])

        # removing prefix -- from remainder args
        script_args = script_args[1:]
        shell_cmd.append(
            "python3 /workspace/scripts/" + args.script + " " + " ".join(script_args)
        )
    else:
        shell_cmd.append("bash")

    cmd.extend([
        args.image_name,
        "/bin/bash",
        "-c",
        " && ".join(shell_cmd),
    ])
    
    print(f"Running Docker container: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Docker run failed.", file=sys.stderr)
        sys.exit(result.returncode)

def parse_args():
    parser = argparse.ArgumentParser(description="Tool for building and running docker images.")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run", help="Run an existing Docker image.")
    run_parser.add_argument("--image-name", help="Docker image to run", default="hyoon11/vllm-dev:20260121_43_py3.12_torch2.9_triton3.5_navi_upstream_6a09612_ubuntu24.04")
    run_parser.add_argument("--script", help="Script to run inside container.")
    run_parser.add_argument("--hf-cache-dir", help="Location of host folder which will be mounter under /root/.cache/huggingface in docker container.", default="./.cache/huggingface")
    run_parser.add_argument("--device", help="/dev/dri/<dir> location of the device", required=True)
    run_parser.add_argument("--device-name", help="Actual name of the device.", required=True)

    return parser.parse_known_args()

def main():
    args, extra_args = parse_args()
    
    if args.command == "run":
        if not args.hf_cache_dir.endswith(".cache/huggingface"):
            print("Huggingface cache dir invalid: must end with .cache/huggingface")
            sys.exit(1)
        run_container(args, extra_args)
    else:
        # TODO: implement build functionality?
        print("Currently only running existing docker images is supported.")
        exit(1)

if __name__ == "__main__":
    main()