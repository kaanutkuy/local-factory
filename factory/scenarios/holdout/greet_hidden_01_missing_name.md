# Holdout scenario: missing --name errors out cleanly

**Satisfied when:**

- Running `python3 greet.py` with no arguments exits with a **non-zero**
  status code, AND
- something is printed to stderr that mentions the missing argument (the
  word `name` should appear in stderr), AND
- nothing resembling `Hello, !` or `Hello, None!` is printed on stdout.

The script must NOT crash with an uncaught traceback (no `Traceback (most
recent call last):` in stderr).
