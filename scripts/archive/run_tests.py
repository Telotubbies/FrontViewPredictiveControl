#!/usr/bin/env python3
"""
Test runner for CARLA MPC system unit tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --specific test_carla_manager  # Run specific test file
"""

import sys
import subprocess
from pathlib import Path

def run_tests(verbose=False, specific_test=None):
    """Run unit tests using pytest."""
    root_dir = Path(__file__).parent
    tests_dir = root_dir / "tests"
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    except ImportError:
        pass
    
    # Specify test file or run all
    if specific_test:
        test_file = tests_dir / f"{specific_test}.py"
        if test_file.exists():
            cmd.append(str(test_file))
        else:
            print(f"Test file not found: {test_file}")
            return False
    else:
        cmd.append(str(tests_dir))
    
    # Add pytest options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "-x",          # Stop on first failure
        "--disable-warnings",  # Disable warnings
    ])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=root_dir, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CARLA MPC unit tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--specific", "-s", help="Run specific test file (without .py extension)")
    
    args = parser.parse_args()
    
    success = run_tests(verbose=args.verbose, specific_test=args.specific)
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
