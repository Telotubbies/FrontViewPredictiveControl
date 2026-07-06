"""Configuration-only tests without external dependencies."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class TestConfigOnly(unittest.TestCase):
    """Test configuration without external dependencies."""

    def test_config_file_exists(self):
        """Test that configuration files exist."""
        config_files = [
            ROOT / "config" / "default.yaml",
            ROOT / "config" / "__init__.py",
        ]

        for config_file in config_files:
            self.assertTrue(config_file.exists(), f"Config file not found: {config_file}")

    def test_config_module_import(self):
        """Test config module import and required attributes."""
        try:
            import config

            # Test that required attributes are set from YAML
            required_attrs = ['MPC_DT', 'MPC_MAX_STEER', 'CAM_W', 'CAM_H']
            for attr in required_attrs:
                self.assertTrue(hasattr(config, attr), f"Missing config attribute: {attr}")

        except Exception as e:
            self.fail(f"Config import failed: {e}")

    def test_basic_python_imports(self):
        """Test that basic Python modules work."""
        import logging
        import time
        import queue
        import threading
        from typing import Optional, Dict, Any, Tuple

        self.assertTrue(hasattr(logging, 'getLogger'))
        self.assertTrue(hasattr(time, 'time'))
        self.assertTrue(hasattr(queue, 'Queue'))
        self.assertTrue(hasattr(threading, 'Thread'))

    def test_file_structure(self):
        """Test that required directories and files exist (src/ layout)."""
        required_dirs = [
            SRC / "control",
            SRC / "perception",
            SRC / "safety",
            SRC / "algorithms",
            SRC / "core",
            SRC / "managers",
            SRC / "utils",
            SRC / "gui",
            ROOT / "config",
            ROOT / "tests" / "unit",
            ROOT / "tests" / "integration",
        ]

        required_files = [
            ROOT / "README.md",
            ROOT / "config" / "default.yaml",
            ROOT / "pyproject.toml",
            ROOT / "STANDARDS.md",
            ROOT / "WORKLOG.md",
        ]

        for dir_name in required_dirs:
            self.assertTrue(dir_name.exists(), f"Directory not found: {dir_name}")
            self.assertTrue(dir_name.is_dir(), f"Not a directory: {dir_name}")

        for file_name in required_files:
            file_path = file_name
            self.assertTrue(file_path.exists(), f"File not found: {file_path}")
            self.assertTrue(file_path.is_file(), f"Not a file: {file_path}")

    def test_python_syntax(self):
        """Test that Python files have valid syntax."""
        python_files = list(SRC.glob("**/*.py"))

        syntax_errors = []

        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")

        if syntax_errors:
            self.fail("Syntax errors found:\n" + "\n".join(syntax_errors))

    def test_import_structure(self):
        """Test that __init__.py files exist where expected (src/ layout)."""
        init_files = [
            SRC / "control" / "__init__.py",
            SRC / "perception" / "__init__.py",
            SRC / "safety" / "__init__.py",
            SRC / "algorithms" / "__init__.py",
            SRC / "core" / "__init__.py",
            SRC / "managers" / "__init__.py",
            SRC / "utils" / "__init__.py",
            SRC / "gui" / "__init__.py",
            SRC / "__init__.py",
            ROOT / "config" / "__init__.py",
        ]

        for init_file in init_files:
            self.assertTrue(init_file.exists(), f"Missing __init__.py: {init_file}")


class TestBugFixes(unittest.TestCase):
    """Test that our bug fixes are in place."""

    def test_carla_manager_fov_fix(self):
        """Test that the FOV fix is present in carla_manager.py."""
        carla_manager_file = SRC / "managers" / "carla_manager.py"

        if not carla_manager_file.exists():
            self.skipTest("carla_manager.py not found")

        with open(carla_manager_file, 'r') as f:
            content = f.read()

        # Check that the fix is present (using config.CAM_FOV_DEG instead of image.attributes)
        self.assertIn("self.config.CAM_FOV_DEG", content,
                      "FOV fix not found - should use self.config.CAM_FOV_DEG")

    def test_pipeline_process_method(self):
        """Test that the process method is present in pipeline.py."""
        pipeline_file = SRC / "pipeline.py"

        if not pipeline_file.exists():
            self.skipTest("pipeline.py not found")

        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn("def process(", content,
                      "process method not found in pipeline.py")
        self.assertIn("self.step(", content,
                      "process method should delegate to step")

    def test_stuck_recovery_should_recover(self):
        """Test that should_recover method is present in StuckRecovery."""
        stuck_recovery_file = SRC / "safety" / "stuck_recovery.py"

        if not stuck_recovery_file.exists():
            self.skipTest("stuck_recovery.py not found")

        with open(stuck_recovery_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn("def should_recover(", content,
                      "should_recover method not found in stuck_recovery.py")

    def test_display_manager_cv2_resize(self):
        """Test that display manager uses cv2.resize instead of np.resize."""
        display_manager_file = SRC / "managers" / "display_manager.py"

        if not display_manager_file.exists():
            self.skipTest("display_manager.py not found")

        with open(display_manager_file, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn("cv2.resize", content,
                      "cv2.resize not found - should use cv2.resize for proper image resizing")


if __name__ == '__main__':
    unittest.main()
