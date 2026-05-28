# Task: greet — a tiny CLI

Add `greet.py` at the repo root. It should be a small Python 3 script that
behaves like a polite greeter you can invoke from the shell.

## What it must do

- Run as `python3 greet.py --name <NAME>` and print exactly one line:
  `Hello, <NAME>!` followed by a newline.
- Run as `python3 greet.py --help` and exit 0 with a usage message that
  mentions the `--name` argument.
- If `--name` is missing or empty, print a helpful error message to stderr
  and exit with a non-zero status. Do NOT print the greeting in that case.

## What it must NOT do

- It must not require any third-party libraries (stdlib only).
- It must not crash with an uncaught exception on any of the cases above.
- It must not write to any file. Stdout / stderr only.

## Style

- Be small. Aim for under ~40 lines.
- One file. No package structure. No tests file (the judge tests behavior,
  not file presence).
