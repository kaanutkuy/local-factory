#!/usr/bin/env python3
"""Turn a string into a URL slug. Stdlib only."""
import re
import sys
import unicodedata


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE)
    return text.strip("-")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    print(slugify(raw))
