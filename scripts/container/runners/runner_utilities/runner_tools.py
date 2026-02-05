"""
running_utils.py - utilities for running inferrence
"""

import time


def generate_and_collect(
    model, duration, iterations, llm, prompts, sampling_params, print_example=True
):
    start = time.monotonic()
    iteration_count = 0
    outputs = []

    def condition():
        if duration:
            return time.monotonic() - start < duration
        elif iterations:
            return iteration_count < iterations
        else:
            raise ValueError(
                "Either duration or iterations must be explicitly provided."
            )

    while condition():
        outputs.extend(llm.generate(prompts, sampling_params))
        iteration_count += 1

    total_duration = time.monotonic() - start

    if print_example:
        print(f"Sample output from {model}: {outputs[0].outputs[0].text}")
    print(f"Total runtime: {total_duration:.2f}s for {iteration_count} iterations.")
    return outputs
