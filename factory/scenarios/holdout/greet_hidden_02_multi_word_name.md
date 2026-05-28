# Holdout scenario: multi-word names are preserved verbatim

**Satisfied when:** running

    python3 greet.py --name "Ada Lovelace"

prints exactly `Hello, Ada Lovelace!` on stdout (with the space preserved
between `Ada` and `Lovelace`) and exits 0.

A solution that strips whitespace, splits on spaces, or only echoes the
first word does NOT satisfy this scenario.
