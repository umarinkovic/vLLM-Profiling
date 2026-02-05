"""
argparse.py - utilities for parsing arguments
"""

import argparse
from pathlib import Path

__all__ = ["parse_and_validate_args"]


def _create_parser(description, resources=False):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--model", help="Name of the model to run.", type=str, required=True
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--duration",
        help="Duration of time (seconds) the model should be running",
        type=int,
    )
    group.add_argument(
        "--iterations", help="Number of iterations the model should run for", type=int
    )
    # TODO: rework so all prompts are singular file with keys according to model type
    parser.add_argument(
        "--prompts-path",
        help="The path of the prompts .yaml file.",
        type=Path,
        required=True,
    )

    if resources:
        parser.add_argument(
            "--resources-path",
            help="Path to the resources referenced in the prompts.",
            type=Path,
            required=True,
        )

    return parser


def _validate_args(args):
    # TODO: revisit if validation necessary
    return


def parse_and_validate_args(description, resources=False, argv=None):
    parser = _create_parser(description=description, resources=resources)
    args, _ = parser.parse_known_args(argv)
    _validate_args(args)
    return args
