"""
Clean Configuration Module - จัดการการตั้งค่าแบบ centralized และ clean

Features:
- Load from YAML config
- Type validation
- Default fallback values
- Environment variable support
- Configuration validation
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import logging
import os
import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration related errors"""
    pass


class Config:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent
        self.config_path = config_path or self.project_root / "config.yaml"
        
        # Configuration data
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        
        # Keys that should be converted from list to tuple
        self._tuple_keys = {
            "RAW_MASK_DILATE_KERNEL", "RAW_MASK_CLOSE_KERNEL", "RAW_MASK_CLOSE_KERNEL2",
            "MASK_DILATE_VERTICAL", "MASK_CLOSE_KERNEL_LARGE",
            "BEV_ALONG_ROAD_DILATE_KERNEL",
        }
        
        # Load configuration
        self._load_defaults()
        self._load_config()
        self._validate_config()
    
    def _load_defaults(self) -> None:
        """Load default fallback values"""
        self._defaults = {
            # Camera defaults
            "CAM_FOV_DEG": 110.0,
            "CAM_W": 640,
            "CAM_H": 480,
            "CTE_TO_METERS": 3.5,
            "UNET_INPUT_W": 640,
            "UNET_INPUT_H": 480,
            "UNET_INPUT_W_INFER": 320,
            "UNET_INPUT_H_INFER": 240,
            "TARGET_SPEED_KMH": 25.0,
            
            # Waypoint defaults
            "NUM_WP": 40,
            "WP_STEP": 2.0,
            "WP_LOOKAHEAD": 8.0,
            
            # Fusion defaults
            "WP_CTE_WEIGHT": 0.80,
            "UNET_CENTER_WEIGHT": 0.20,
            "LANE_CONF_THRESHOLD": 0.45,
            "WP_DIRECTION_WEIGHT": 1.0,
            
            # Lane defaults
            "ROI_BOTTOM_RATIO": 0.55,
            "ROI_TOP_PENALTY_RATIO": 0.35,
            "ROAD_TOP_RATIO": 0.0,
            "BEV_LANE_SEARCH_TOP_RATIO": 0.1,
            "BEV_HISTOGRAM_WEIGHT_FAR": 1.5,
            "BEV_HISTOGRAM_WEIGHT_NEAR": 0.5,
            "BEV_WARP_INTERPOLATION": "nearest",
            "BEV_RIGHTMOST_PEAK": True,
            "BEV_RIGHT_PEAK_MIN_RATIO": 0.25,
            "BEV_ALONG_ROAD_DILATE_KERNEL": (3, 5),
            "LANE_CONF_ROI_TOP_RATIO": 0.30,
            "LANE_CONF_CONTINUITY_ALPHA": 0.4,
            "LANE_CONTINUITY_MIN_AREA_PX": 50,
            "RAW_MASK_DILATE_KERNEL": (3, 3),
            "RAW_MASK_CLOSE_KERNEL": (5, 5),
            "RAW_MASK_CLOSE_KERNEL2": (7, 7),
            "MASK_DILATE_VERTICAL": (3, 14),
            "MASK_CLOSE_KERNEL_LARGE": (15, 8),
            "ROAD_MARGIN_PX": 8,
            "MIN_LANE_WIDTH_PX": 12,
            "MAX_LANE_WIDTH_PX": 280,
            "CENTER_LANE_WIDTH_RATIO": 0.65,
            "LANE_MEMORY_MARGIN_PX": 55,
            "LANE_EXTEND_ROWS_UP": 72,
            "LANE_FILL_STEP": 1,
            "DASHED_LANE_MIN_PIXELS": 5,
            "MIN_LANE_WIDTH_M": 2.0,
            "MAX_LANE_WIDTH_M": 5.5,
            "LANE_OVERLAY_ROAD_FORM": False,
            "LANE_OVERLAY_STRIP_HALF_WIDTH_M": 0.9,
            "LANE_OVERLAY_STRAIGHT_NEAR_M": 2.0,
            "LANE_FRAME_WIDTH_MIN_M": 2.0,
            "LANE_FRAME_WIDTH_MAX_M": 5.5,
            "LANE_FRAME_WIDTH_NOMINAL_M": 3.5,
            "LANE_FRAME_WIDTH_EMA_ALPHA": 0.6,
            "LANE_OVERLAY_USE_FRAME_WIDTH": False,
            "LANE_OVERLAY_TRAPEZOID": True,
            
            # Reference/MPC defaults
            "CTE_LANE_HALF_WIDTH_M": 0.55,
            "MAX_CURVATURE_REF": 0.035,
            "STEER_SMOOTH_ALPHA": 0.55,
            "STEER_MAX_DELTA_PER_FRAME": 0.12,
            "STEER_RATE_LIMIT_SPEED_KMH": 40.0,
            "STEER_MAX_DELTA_HIGH_SPEED": 0.06,
            "USE_TRAJECTORY_PIPELINE": True,
            "LANE_EMA_ALPHA": 0.38,
            "STATE_EMA_ALPHA": 0.35,
            "REF_PATH_LOOKAHEAD_M": 35.0,
            "REF_PATH_NUM_PTS": 70,
            "REF_PATH_DISPLAY_LOOKAHEAD_M": 200.0,
            "REF_PATH_DISPLAY_NUM_PTS": 200,
            "PLANNING_VIEW_RADIUS_M": 500.0,
            "TURN_LOOKAHEAD_MIN_M": 12.0,
            "TURN_LOOKAHEAD_CURV_THRESHOLD": 0.02,
            "PERCEPTION_PATH_MIN_CONF": 0.5,
            "REF_PATH_LAT_SANITY_M": 5.0,
            "PERCEPTION_PATH_SMOOTH_ALPHA": 0.6,
            "CURVATURE_WP_EMA_ALPHA": 0.5,
            "LANE_HALF_WIDTH_DRAW_M": 0.9,
            "LANE_OVERLAY_DISPLAY_MARGIN_M": 0.05,
            "LANE_OVERLAY_MIN_DISPLAY_WIDTH_M": 1.2,
            "LANE_OVERLAY_MAX_WIDTH_DRAW_M": 6.0,
            "LOOKAHEAD_MIN_M": 8.0,
            "LOOKAHEAD_TIME_S": 2.2,
            "FUSION_CONF_HIGH": 0.85,
            "FUSION_CONF_LOW": 0.55,
            "FUSION_CURVATURE_UNET_REDUCE_THRESHOLD": 0.025,
            "FUSION_CURVATURE_UNET_REDUCE_SCALE": 0.02,
            
            # MPC defaults
            "MPC_N": 10,
            "MPC_DT": 0.1,
            "MPC_L": 2.875,
            "MPC_MAX_STEER": 0.7,
            "MPC_MAX_STEER_RATE": 0.15,
            "MPC_MAX_ACCEL": 3.0,
            "MPC_MIN_ACCEL": -5.0,
            "MPC_W_CTE": 350.0,
            "MPC_W_HEADING": 85.0,
            "MPC_W_VEL": 5.0,
            "MPC_W_STEER": 80.0,
            "MPC_W_ACCEL": 10.0,
            "MPC_W_STEER_RATE": 1000.0,
            "MPC_W_ACCEL_RATE": 10.0,
            "MPC_W_STEER_JERK": 180.0,
            "MPC_USE_ADAPTIVE_WEIGHTS": True,
            
            # View defaults
            "REF_PATH_DISPLAY_EMA_ALPHA": 0.82,
            "BEV_PX_PER_M": 35,
            "BEV_W": 450,
            "BEV_H": 320,
            "VIEW3D_W": 350,
            "VIEW3D_H": 200,
            "BEV_WIN_W": 350,
            "BEV_WIN_H": 200,
            "PANEL_W": 450,
            "PANEL_H": 320,
            
            # Runtime defaults
            "USE_PERCEPTION_THREAD": True,
            "CONTROL_HZ": 10,
            "STALE_PERCEPTION_S": 0.2,
            "PERCEPTION_SKIP_FRAME": 2,
            "PERCEPTION_LIGHTWEIGHT_VIS": False,
            "MAIN_LOOP_SLEEP_S": 0.01,
            "DASHBOARD_DRAW_EVERY_N": 1,
            "DISPLAY_PATH_CACHE_INTERVAL": 5,
            
            # Safety defaults
            "SAFETY_MAX_STEER_RAD": 0.45,
            
            # UX Colors
            "UXColors": {
                "BACKGROUND": (20, 20, 30),
                "TEXT": (200, 200, 200),
                "SUCCESS": (0, 255, 0),
                "WARNING": (255, 255, 0),
                "ERROR": (255, 0, 0),
                "LANE": (0, 255, 0),
                "BOUNDARY": (255, 255, 255),
            }
        }
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self._config = self._defaults.copy()
                return
            
            with open(self.config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                logger.warning("Empty config file, using defaults")
                self._config = self._defaults.copy()
                return
            
            # Flatten nested structure
            flat_config = {}
            for section, values in yaml_data.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        flat_config[k] = v
                else:
                    flat_config[section] = values
            
            # Apply defaults and loaded config
            self._config = self._defaults.copy()
            self._config.update(flat_config)
            
            # Convert lists to tuples where needed
            for key in self._tuple_keys:
                if key in self._config and isinstance(self._config[key], list):
                    self._config[key] = tuple(self._config[key])
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}, using defaults")
            self._config = self._defaults.copy()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""
        env_prefix = "CARLA_MPC_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):]
                try:
                    # Try to parse as number, otherwise keep as string
                    if value.lower() in ('true', 'false'):
                        self._config[config_key] = value.lower() == 'true'
                    elif value.isdigit():
                        self._config[config_key] = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        self._config[config_key] = float(value)
                    else:
                        self._config[config_key] = value
                except ValueError:
                    self._config[config_key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        errors = []
        
        # Validate numeric ranges
        numeric_ranges = {
            "TARGET_SPEED_KMH": (0, 200),
            "CAM_W": (100, 1920),
            "CAM_H": (100, 1080),
            "CONTROL_HZ": (1, 100),
            "MPC_MAX_STEER": (0, 1.57),
            "SAFETY_MAX_STEER_RAD": (0, 1.57),
        }
        
        for key, (min_val, max_val) in numeric_ranges.items():
            if key in self._config:
                value = self._config[key]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    errors.append(f"{key} must be between {min_val} and {max_val}, got {value}")
        
        # Validate file paths
        if "MODEL_PATH" in self._config:
            model_path = Path(self._config["MODEL_PATH"])
            if not model_path.exists():
                errors.append(f"Model file not found: {model_path}")
        
        if errors:
            raise ConfigError("Configuration validation failed:\n" + "\n".join(errors))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all configuration values for a section (based on key prefixes)"""
        section_config = {}
        prefix = section.upper() + "_"
        
        for key, value in self._config.items():
            if key.startswith(prefix):
                section_key = key[len(prefix):]
                section_config[section_key] = value
        
        return section_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save current configuration to YAML file"""
        save_path = path or self.config_path
        
        # Group by sections
        sections = {}
        for key, value in self._config.items():
            if "_" in key:
                section, subkey = key.split("_", 1)
                if section not in sections:
                    sections[section] = {}
                sections[section][subkey] = value
            else:
                sections[key] = value
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(sections, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access for configuration values"""
        if key in self._config:
            return self._config[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        """Allow attribute-style setting for configuration values"""
        if key.startswith('_') or key in {'project_root', 'config_path', '_config', '_defaults', '_tuple_keys'}:
            super().__setattr__(key, value)
        else:
            self._config[key] = value


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config(config_path: Optional[Path] = None) -> Config:
    """Reload configuration from file"""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance


# Export configuration as module-level attributes for backward compatibility
_config = get_config()

# Create module-level attributes for backward compatibility
for key, value in _config.to_dict().items():
    globals()[key] = value
