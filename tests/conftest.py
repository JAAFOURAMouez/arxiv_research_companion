"""Make the repo root importable and register markers (see pyproject.toml)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
