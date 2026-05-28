# Holdout scenario: --help exits 0 and explains --name

**Satisfied when:** running

    python3 greet.py --help

exits with status 0, and the stdout (or stderr — wherever the script
chose to print usage) contains both the word `--name` and the word
`usage` (or `Usage`).

The script must not raise an exception when `--help` is passed.
