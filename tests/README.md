# CARLA MPC Unit Tests

This directory contains unit tests for the CARLA MPC system.

## Test Files

### Basic Tests (No External Dependencies)
- **`test_config_only.py`** - Basic functionality tests that verify:
  - Configuration file existence and loading
  - File structure and import organization
  - Python syntax validation across all files
  - Bug fixes are present in the codebase

### Component Tests (Require Dependencies)
- **`test_carla_manager.py`** - Tests for CarlaManager:
  - Camera callback image conversion
  - FOV configuration handling (fixed bug)
  - Queue overflow handling
  - BGR to RGB color conversion

- **`test_pipeline.py`** - Tests for LKAPipeline:
  - Process method existence and functionality
  - Integration with trajectory pipeline
  - Error handling

- **`test_stuck_recovery.py`** - Tests for StuckRecovery:
  - should_recover method functionality
  - State transitions during recovery
  - Edge case handling

- **`test_display_manager.py`** - Tests for DisplayManager:
  - Image rendering with cv2.resize (fixed bug)
  - Color validation and error handling
  - GUI exception handling

- **`test_integration.py`** - Integration tests:
  - End-to-end pipeline testing
  - Component interaction verification
  - Configuration consistency

## Running Tests

### With Virtual Environment (Recommended)
```bash
# Activate environment
source activate_venv.sh

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_config_only.py -v

# Run specific test method
python -m pytest tests/test_carla_manager.py::TestCarlaManager::test_camera_callback_image_conversion -v
```

### Alternative Test Runners
```bash
# Simple test runner (no dependencies)
python run_unittests.py --verbose

# Pytest with coverage (if available)
python -m pytest tests/ --cov=. --cov-report=html
```

## Bug Fixes Verified by Tests

The tests specifically verify these bug fixes:

1. **Camera FOV Bug**: Fixed `image.attributes.get('fov')` to use `self.config.CAM_FOV_DEG`
2. **Image Rendering Bug**: Fixed `np.resize()` to use `cv2.resize()` for proper aspect ratio
3. **Missing Methods**: Added `process()` method to LKAPipeline and `should_recover()` method to StuckRecovery
4. **Color Validation**: Added proper validation for confidence color values

## Test Coverage

- ✅ Configuration loading and validation
- ✅ Camera image processing and conversion
- ✅ Pipeline integration and error handling
- ✅ Stuck recovery state management
- ✅ Display rendering and color handling
- ✅ File structure and syntax validation
- ✅ Bug fix verification

## Adding New Tests

When adding new tests:

1. Use the existing test structure and naming conventions
2. Mock external dependencies (CARLA, numpy, torch) when needed
3. Test both happy path and error conditions
4. Verify bug fixes remain in place
5. Add tests to `test_config_only.py` for dependency-free checks

## Dependencies

- **Basic tests**: No external dependencies required
- **Component tests**: Requires virtual environment with:
  - numpy
  - torch
  - opencv-python
  - pygame
  - pytest

## Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source activate_venv.sh

# Install missing dependencies if needed
pip install pytest numpy torch opencv-python pygame
```

### Test Failures
- Check that virtual environment is activated
- Verify all mock objects are properly configured
- Ensure test data (images, configs) is accessible
- Check for recent code changes that might affect test expectations
