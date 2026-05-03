#!/usr/bin/env python3
"""Run all ADAS v2 unit tests."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

if __name__ == "__main__":
    # Run all tests in this directory
    test_dir = Path(__file__).parent
    exit_code = pytest.main([
        str(test_dir),
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])
    sys.exit(exit_code)
