#!/usr/bin/env python3
"""A tiny polite greeter CLI."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print a polite greeting.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name of the person to greet.",
    )
    args = parser.parse_args(argv)

    if not args.name:
        print("error: --name must not be empty", file=sys.stderr)
        return 2

    print(f"Hello, {args.name}!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
