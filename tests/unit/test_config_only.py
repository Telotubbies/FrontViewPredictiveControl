"""Configuration-only tests without external dependencies."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestConfigOnly(unittest.TestCase):
    """Test configuration without external dependencies."""

    def test_config_file_exists(self):
        """Test that configuration files exist."""
        config_files = [
            ROOT / "config.yaml",
            ROOT / "config_clean.py",
        ]

        for config_file in config_files:
            self.assertTrue(config_file.exists(), f"Config file not found: {config_file}")

    def test_config_clean_basic_import(self):
        """Test basic config_clean import without dependencies."""
        try:
            # Try to import just the Config class
            sys.path.insert(0, str(ROOT))

            # Mock the yaml module to avoid dependency
            import types
            yaml_mock = types.ModuleType('yaml')
            yaml_mock.safe_load = lambda x: {}
            sys.modules['yaml'] = yaml_mock

            from config_clean import Config

            # Test basic instantiation
            config = Config()

            # Test that default values are set
            self.assertTrue(hasattr(config, '_defaults'))
            self.assertIsInstance(config._defaults, dict)

        except Exception as e:
            self.fail(f"Config import failed: {e}")

    def test_basic_python_imports(self):
        """Test that basic Python modules work."""
        # Test standard library imports
        import logging
        import time
        import queue
        import threading
        from typing import Optional, Dict, Any, Tuple

        # Test that they work
        self.assertTrue(hasattr(logging, 'getLogger'))
        self.assertTrue(hasattr(time, 'time'))
        self.assertTrue(hasattr(queue, 'Queue'))
        self.assertTrue(hasattr(threading, 'Thread'))

    def test_file_structure(self):
        """Test that required directories and files exist."""
        required_dirs = [
            "managers",
            "safety",
            "utils",
            "tests"
        ]

        required_files = [
            "README.md",
            "config.yaml"
        ]

        for dir_name in required_dirs:
            dir_path = ROOT / dir_name
            self.assertTrue(dir_path.exists(), f"Directory not found: {dir_path}")
            self.assertTrue(dir_path.is_dir(), f"Not a directory: {dir_path}")

        for file_name in required_files:
            file_path = ROOT / file_name
            self.assertTrue(file_path.exists(), f"File not found: {file_path}")
            self.assertTrue(file_path.is_file(), f"Not a file: {file_path}")

    def test_python_syntax(self):
        """Test that Python files have valid syntax."""
        python_files = list(ROOT.glob("**/*.py"))

        syntax_errors = []

        for py_file in python_files:
            # Skip test files and __pycache__
            if "test" in py_file.name or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")

        if syntax_errors:
            self.fail("Syntax errors found:\n" + "\n".join(syntax_errors))

    def test_import_structure(self):
        """Test that import structure is consistent."""
        # Test that __init__.py files exist where expected
        init_files = [
            ROOT / "managers" / "__init__.py",
            ROOT / "safety" / "__init__.py",
            ROOT / "utils" / "__init__.py",
            ROOT / "tests" / "__init__.py",
        ]

        for init_file in init_files:
            self.assertTrue(init_file.exists(), f"Missing __init__.py: {init_file}")


class TestBugFixes(unittest.TestCase):
    """Test that our bug fixes are in place."""

    def test_carla_manager_fov_fix(self):
        """Test that the FOV fix is present in carla_manager.py."""
        carla_manager_file = ROOT / "managers" / "carla_manager.py"

        if not carla_manager_file.exists():
            self.skipTest("carla_manager.py not found")

        with open(carla_manager_file, 'r') as f:
            content = f.read()

        # Check that the fix is present (using config.CAM_FOV_DEG instead of image.attributes)
        self.assertIn("self.config.CAM_FOV_DEG", content,
                      "FOV fix not found - should use self.config.CAM_FOV_DEG")

        # Check that the old buggy code is not present
        self.assertNotIn("image.attributes.get('fov'", content,
                        "Old buggy code still present - should not use image.attributes")

    def test_pipeline_process_method(self):
        """Test that the process method is present in pipeline.py."""
        pipeline_file = ROOT / "pipeline.py"

        if not pipeline_file.exists():
            self.skipTest("pipeline.py not found")

        with open(pipeline_file, 'r') as f:
            content = f.read()

        # Check that process method exists
        self.assertIn("def process(", content,
                      "process method not found in pipeline.py")

        # Check that it delegates to step
        self.assertIn("self.step(", content,
                      "process method should delegate to step")

    def test_stuck_recovery_should_recover(self):
        """Test that should_recover method is present in StuckRecovery."""
        stuck_recovery_file = ROOT / "safety" / "stuck_recovery.py"

        if not stuck_recovery_file.exists():
            self.skipTest("stuck_recovery.py not found")

        with open(stuck_recovery_file, 'r') as f:
            content = f.read()

        # Check that should_recover method exists
        self.assertIn("def should_recover(", content,
                      "should_recover method not found in stuck_recovery.py")

    def test_display_manager_cv2_resize(self):
        """Test that display manager uses cv2.resize instead of np.resize."""
        display_manager_file = ROOT / "managers" / "display_manager.py"

        if not display_manager_file.exists():
            self.skipTest("display_manager.py not found")

        with open(display_manager_file, 'r') as f:
            content = f.read()

        # Check that cv2.resize is used
        self.assertIn("cv2.resize", content,
                      "cv2.resize not found - should use cv2.resize for proper image resizing")

        # Check that np.resize is not used for image resizing
        lines = content.split('\n')
        for line in lines:
            if 'np.resize' in line and 'frame' in line.lower():
                self.fail("Found np.resize used for frame resizing - should use cv2.resize")


if __name__ == '__main__':
    unittest.main()
