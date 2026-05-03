#!/usr/bin/env python3
"""
Simple unittest runner for CARLA MPC system.

Usage:
    python run_unittests.py              # Run all tests
    python run_unittests.py --verbose    # Run with verbose output
"""

import sys
import unittest
from pathlib import Path

def discover_and_run_tests(verbose=False):
    """Discover and run all unittest tests."""
    # Get the tests directory
    tests_dir = Path(__file__).parent / "tests"
    
    # Discover all test files
    loader = unittest.TestLoader()
    start_dir = str(tests_dir)
    pattern = "test_*unittest.py"
    
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Create test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    
    # Run tests
    print(f"Running unittests from {tests_dir}")
    print("=" * 50)
    
    result = runner.run(suite)
    
    # Print summary
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CARLA MPC unittests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    success = discover_and_run_tests(verbose=args.verbose)
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
