# Task: slugify — turn a string into a URL slug

Add `slugify.py` at the repo root. It is a small Python 3 CLI that takes a
single positional string argument and prints a URL slug to stdout.

## What it must do

- Run as `python3 slugify.py "<text>"` and print the resulting slug
  followed by a newline.
- The output slug must be lowercase.
- Spaces should be converted to hyphens.
- Exit 0 on success.

## What it must NOT do

- Require any third-party libraries (stdlib only).
- Crash with an uncaught exception on reasonable input.
- Write to any file. Stdout only.

## Style

- One file, at the repo root, named `slugify.py`.
- Aim for under 30 lines.
- No tests file. The judge tests behavior, not file presence.
