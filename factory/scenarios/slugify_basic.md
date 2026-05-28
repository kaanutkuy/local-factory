# Visible scenarios: basic slugification

## Scenario 1: simple two-word title

**Satisfied when:** running

    python3 slugify.py "Hello World"

prints exactly `hello-world` followed by a newline, and exits 0.

## Scenario 2: three-word title

**Satisfied when:** running

    python3 slugify.py "Foo Bar Baz"

prints exactly `foo-bar-baz` followed by a newline, and exits 0.
