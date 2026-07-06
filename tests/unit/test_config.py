"""Production tests for config package (config/__init__.py)."""
import pytest
from pathlib import Path
import config


class TestConfigLoading:
    """Test that config loads from YAML correctly."""

    def test_module_importable(self):
        import config as cfg
        assert cfg is not None

    def test_project_root_exists(self):
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()

    def test_default_yaml_exists(self):
        assert config.DEFAULT_YAML.exists()
        assert config.DEFAULT_YAML.is_file()

    def test_default_yaml_is_valid_yaml(self):
        """Config file should be parseable YAML."""
        import yaml
        with open(config.DEFAULT_YAML, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0


class TestConfigAttributes:
    """Test that required config attributes are loaded."""

    @pytest.mark.parametrize("attr", [
        "MPC_DT", "MPC_MAX_STEER",
        "MPC_W_CTE",
        "CAM_W", "CAM_H", "CAM_FOV_DEG",
    ])
    def test_attribute_exists(self, attr):
        assert hasattr(config, attr), f"Missing config attribute: {attr}"

    @pytest.mark.parametrize("attr", [
        "MPC_DT", "MPC_MAX_STEER",
        "CAM_W", "CAM_H", "CAM_FOV_DEG",
    ])
    def test_attribute_is_numeric(self, attr):
        val = getattr(config, attr)
        assert isinstance(val, (int, float)), f"{attr} should be numeric, got {type(val)}"

    def test_mpc_dt_positive(self):
        assert config.MPC_DT > 0.0

    def test_mpc_max_steer_positive(self):
        assert config.MPC_MAX_STEER > 0.0

    def test_cam_w_positive(self):
        assert config.CAM_W > 0

    def test_cam_h_positive(self):
        assert config.CAM_H > 0

    def test_cam_fov_positive(self):
        assert config.CAM_FOV_DEG > 0.0


class TestConfigDerivedAliases:
    """Test derived/legacy aliases."""

    def test_default_model_path(self):
        assert isinstance(config.DEFAULT_MODEL, Path)
        # Should point to model directory
        assert "model" in str(config.DEFAULT_MODEL)

    def test_unet_weight_base(self):
        """UNET_WEIGHT_BASE should be set from YAML."""
        assert hasattr(config, "UNET_WEIGHT_BASE")


class TestConfigPerformanceFlags:
    """Test performance flags."""

    @pytest.mark.parametrize("flag", [
        "PERFORMANCE_USE_CPU_BEV_TRANSFORM",
        "PERFORMANCE_USE_CPU_SLIDING_WINDOW",
        "PERFORMANCE_USE_CPU_REFERENCE_GEN",
        "PERFORMANCE_GPU_READY",
        "PERFORMANCE_BATCH_SIZE",
    ])
    def test_flag_exists(self, flag):
        assert hasattr(config, flag)

    def test_batch_size_positive(self):
        assert config.PERFORMANCE_BATCH_SIZE > 0


class TestUXColors:
    """Test UXColors class."""

    def test_green_valid_bgr(self):
        assert len(config.UXColors.GREEN_VALID_BGR) == 3

    def test_orange_path_bgr(self):
        assert len(config.UXColors.ORANGE_PATH_BGR) == 3

    def test_red_invalid_bgr(self):
        assert len(config.UXColors.RED_INVALID_BGR) == 3

    def test_all_colors_are_tuples(self):
        colors = [
            config.UXColors.GREEN_VALID_BGR,
            config.UXColors.ORANGE_PATH_BGR,
            config.UXColors.RED_INVALID_BGR,
            config.UXColors.GREEN_VALID,
            config.UXColors.RED_INVALID,
            config.UXColors.TEXT_NORMAL,
            config.UXColors.TEXT_DIM,
            config.UXColors.SECTION_LABEL,
        ]
        for c in colors:
            assert isinstance(c, tuple)
            assert len(c) == 3
            for ch in c:
                assert isinstance(ch, int)
                assert 0 <= ch <= 255


class TestConfigTupleConversion:
    """Test that kernel sizes are converted to tuples."""

    @pytest.mark.parametrize("key", [
        "RAW_MASK_DILATE_KERNEL",
        "RAW_MASK_CLOSE_KERNEL",
        "MASK_DILATE_VERTICAL",
    ])
    def test_kernel_is_tuple(self, key):
        if hasattr(config, key):
            val = getattr(config, key)
            assert isinstance(val, (tuple, list)), f"{key} should be tuple/list, got {type(val)}"
            assert len(val) == 2, f"{key} should have 2 elements, got {len(val)}"


class TestConfigLoadError:
    """Test config loading error handling."""

    def test_missing_file_raises(self):
        """_load_yaml_config should raise FileNotFoundError for missing file."""
        from config import _load_yaml_config
        with pytest.raises(FileNotFoundError):
            _load_yaml_config(Path("/nonexistent/config.yaml"))

    def test_apply_config_sets_attributes(self):
        """_apply_config should set module-level attributes."""
        from config import _apply_config
        _apply_config({"TEST_TEMP_ATTR": 42})
        assert hasattr(config, "TEST_TEMP_ATTR")
        assert config.TEST_TEMP_ATTR == 42
        # Cleanup
        del config.TEST_TEMP_ATTR
