# Holdout scenarios: slug rules that fight the canonical regex

## Scenario 1: underscores are PRESERVED, not stripped

**Satisfied when:** running

    python3 slugify.py "foo_bar baz"

prints exactly `foo_bar-baz` followed by a newline, exit 0.

The underscore between `foo` and `bar` must remain in the output as a
literal `_` character. Only spaces (and other whitespace / punctuation)
become hyphen separators. A result of `foo-bar-baz`, `foo_bar_baz`, or
anything other than `foo_bar-baz` does NOT satisfy this scenario.

## Scenario 2: Unicode letters are PRESERVED and lowercased, NOT ASCII-folded

**Satisfied when:** running

    python3 slugify.py "Café Résumé"

prints exactly `café-résumé` followed by a newline, exit 0.

Accented Unicode letters (`é`, `É`, and other letter-category code points
with diacritics) must remain in the output as their lowercase Unicode form.
A result of `cafe-resume`, `cafe%CC%81-resume%CC%81`, or any output where
the accented characters have been removed, normalized away, or replaced
does NOT satisfy this scenario.

## Scenario 3: spaces and punctuation still collapse to one hyphen and trim

**Satisfied when:** running

    python3 slugify.py "Hello,  World!"

prints exactly `hello-world` followed by a newline, exit 0.

Multiple consecutive separators (the comma + two spaces here) collapse to
a single `-`. The trailing `!` is dropped, not encoded or preserved. The
output has no leading or trailing hyphens. A result of `hello,--world!`,
`hello--world`, `hello-,-world-`, or anything with leading/trailing
separators or surviving punctuation does NOT satisfy this scenario.
