"""The fixture target's own tests (stdlib only, so gates need no extra installs)."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


class GreetTest(unittest.TestCase):
    def test_greets(self) -> None:
        result = run("greet.py", "--name", "Ada Lovelace")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Hello, Ada Lovelace!\n")

    def test_missing_name_fails(self) -> None:
        result = run("greet.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("name", result.stderr)


class SlugifyTest(unittest.TestCase):
    def test_slugifies(self) -> None:
        result = run("slugify.py", "Hello,  World!")
        self.assertEqual(result.stdout, "hello-world\n")


if __name__ == "__main__":
    unittest.main()
