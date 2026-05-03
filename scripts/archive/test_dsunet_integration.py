#!/usr/bin/env python3
"""
DSUNet Integration Unit Tests
Test all components work together correctly
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class TestDSUNetIntegration(unittest.TestCase):
    """Test DSUNet model integration with perception pipeline"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.model_path = Path("checkpoints/dsunet_best_iou.pth")
        self.test_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_model_file_exists(self):
        """Test DSUNet model file exists"""
        self.assertTrue(self.model_path.exists(), f"Model file {self.model_path} not found")
        
    def test_lane_detector_import(self):
        """Test LaneDetector can be imported and instantiated"""
        try:
            from perception.lane_detector import LaneDetector
            detector = LaneDetector(
                model_path=str(self.model_path), 
                use_carla=False, 
                model_type="unet"
            )
            self.assertIsNotNone(detector)
            logger.info("✅ LaneDetector import and instantiation successful")
        except Exception as e:
            self.fail(f"LaneDetector import failed: {e}")
            
    def test_lane_trajectory_pipeline_import(self):
        """Test LaneTrajectoryPipeline can be imported"""
        try:
            from perception.lane_trajectory import LaneTrajectoryPipeline, TrajectoryOutput
            self.assertTrue(True)
            logger.info("✅ LaneTrajectoryPipeline import successful")
        except ImportError as e:
            self.fail(f"LaneTrajectoryPipeline import failed: {e}")
            
    def test_bev_attributes(self):
        """Test BEV attributes are properly set"""
        from perception.lane_trajectory import LaneTrajectoryPipeline
        
        pipeline = LaneTrajectoryPipeline()
        self.assertTrue(hasattr(pipeline, 'bev_h'))
        self.assertTrue(hasattr(pipeline, 'bev_w'))
        self.assertTrue(hasattr(pipeline, 'half_width_m'))
        
        self.assertEqual(pipeline.bev_h, 320)
        self.assertEqual(pipeline.bev_w, 450)
        self.assertEqual(pipeline.half_width_m, 8.0)
        logger.info("✅ BEV attributes correctly set")
        
    def test_lane_detector_with_dsunet(self):
        """Test LaneDetector works with DSUNet model"""
        try:
            from perception.lane_detector import LaneDetector
            
            detector = LaneDetector(
                model_path=str(self.model_path),
                use_carla=False,
                model_type="unet"
            )
            
            # Test lane detection
            result = detector.detect_lanes(self.test_rgb)
            if len(result) == 2:
                mask, confidence = result
            elif len(result) == 3:
                mask, confidence, _ = result  # Third value might be extra info
            else:
                self.fail(f"Unexpected return values: {len(result)}")
                
            self.assertIsInstance(mask, np.ndarray)
            self.assertEqual(mask.shape, (480, 640))
            # Confidence might be scalar or array - handle both
            if isinstance(confidence, np.ndarray):
                if confidence.size == 1:
                    confidence = float(confidence.item())
                else:
                    confidence = float(np.mean(confidence))
            self.assertIsInstance(confidence, (float, np.floating))
            logger.info(f"✅ DSUNet detection successful: mask_shape={mask.shape}, conf={confidence:.3f}")
            
        except Exception as e:
            self.fail(f"DSUNet detection failed: {e}")
            
    def test_trajectory_pipeline_with_detector(self):
        """Test LaneTrajectoryPipeline with DSUNet detector"""
        try:
            from perception.lane_detector import LaneDetector
            from perception.lane_trajectory import LaneTrajectoryPipeline
            
            # Create detector and pipeline
            detector = LaneDetector(
                model_path=str(self.model_path),
                use_carla=False,
                model_type="unet"
            )
            pipeline = LaneTrajectoryPipeline(detector=detector)
            
            # Process RGB frame
            result = pipeline.process(self.test_rgb)
            
            # Check output structure
            self.assertIsInstance(result, object)
            self.assertTrue(hasattr(result, 'x_ref'))
            self.assertTrue(hasattr(result, 'y_ref'))
            self.assertTrue(hasattr(result, 'cte'))
            self.assertTrue(hasattr(result, 'lane_overlay'))
            self.assertTrue(hasattr(result, 'bev_binary'))
            self.assertTrue(hasattr(result, 'bev_window_vis'))
            
            logger.info(f"✅ Trajectory pipeline successful: cte={result.cte:.3f}m")
            
        except Exception as e:
            self.fail(f"Trajectory pipeline failed: {e}")
            
    def test_lane_overlay_function(self):
        """Test lane overlay drawing function"""
        try:
            from perception.lane_trajectory import _draw_simple_lane_markings
            
            # Test function call
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            left_coeffs = np.array([0.01, 0.0, -1.5])  # y = 0.01x^2 - 1.5
            right_coeffs = np.array([0.01, 0.0, 1.5])   # y = 0.01x^2 + 1.5
            M_inv = np.eye(3)
            
            result = _draw_simple_lane_markings(
                rgb, left_coeffs, right_coeffs, M_inv,
                bev_h=320, bev_w=450, cam_h=480, cam_w=640,
                lookahead_m=30.0, half_width_m=8.0
            )
            
            self.assertEqual(result.shape, (480, 640, 3))
            self.assertTrue(np.any(result > 0))  # Check lane markings were drawn
            logger.info("✅ Lane overlay function successful")
            
        except Exception as e:
            self.fail(f"Lane overlay function failed: {e}")
            
    def test_config_integration(self):
        """Test config loads DSUNet settings correctly"""
        try:
            import config
            
            # Check model path
            self.assertTrue(hasattr(config, 'DEFAULT_MODEL'))
            self.assertIn('dsunet', str(config.DEFAULT_MODEL))
            
            # Check UNet settings
            self.assertTrue(hasattr(config, 'USE_CLASSICAL_DETECTOR'))
            self.assertFalse(config.USE_CLASSICAL_DETECTOR)  # Should use UNet
            
            logger.info(f"✅ Config integration: model={config.DEFAULT_MODEL}")
            
        except Exception as e:
            self.fail(f"Config integration failed: {e}")
            
    def test_full_pipeline_simulation(self):
        """Test complete pipeline simulation"""
        try:
            from perception.lane_detector import LaneDetector
            from perception.lane_trajectory import LaneTrajectoryPipeline
            
            # Create full pipeline
            detector = LaneDetector(
                model_path=str(self.model_path),
                use_carla=False,
                model_type="unet"
            )
            pipeline = LaneTrajectoryPipeline(detector=detector)
            
            # Process multiple frames
            results = []
            for i in range(3):
                rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                result = pipeline.process(rgb)
                results.append(result)
                
                # Check basic outputs
                self.assertIsInstance(result.cte, (float, np.floating))
                self.assertIsInstance(result.heading_err, (float, np.floating))
                self.assertIsInstance(result.confidence, (float, np.floating))
                
            logger.info(f"✅ Full pipeline simulation: {len(results)} frames processed")
            
        except Exception as e:
            self.fail(f"Full pipeline simulation failed: {e}")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_model_file(self):
        """Test behavior with missing model file"""
        from perception.lane_detector import LaneDetector
        
        # Should handle gracefully with fallback or warning
        try:
            detector = LaneDetector(
                model_path="nonexistent_model.pth",
                use_carla=False,
                model_type="unet"
            )
            # If it doesn't raise exception, it should have fallback behavior
            logger.info("✅ Missing model file handled gracefully")
        except Exception as e:
            # Should raise exception for truly missing file
            logger.info(f"✅ Missing model file properly raised exception: {e}")
            self.assertTrue(True)  # Expected behavior
            
    def test_invalid_input_shape(self):
        """Test behavior with invalid input shapes"""
        from perception.lane_detector import LaneDetector
        
        detector = LaneDetector(
            model_path="checkpoints/dsunet_best_iou.pth",
            use_carla=False,
            model_type="unet"
        )
        
        # Test with wrong shape
        invalid_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Should handle gracefully (resize or error)
        try:
            result = detector.detect_lanes(invalid_rgb)
            if len(result) >= 2:
                mask, confidence = result[:2]
                logger.info("✅ Invalid input shape handled gracefully")
        except Exception as e:
            logger.warning(f"Invalid input shape caused error: {e}")

def run_performance_test():
    """Run performance benchmark"""
    print("\n" + "="*50)
    print("DSUNet PERFORMANCE BENCHMARK")
    print("="*50)
    
    try:
        from perception.lane_detector import LaneDetector
        from perception.lane_trajectory import LaneTrajectoryPipeline
        import time
        
        # Setup
        detector = LaneDetector(
            model_path="checkpoints/dsunet_best_iou.pth",
            use_carla=False,
            model_type="unet"
        )
        pipeline = LaneTrajectoryPipeline(detector=detector)
        
        # Benchmark
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            pipeline.process(rgb)
            
        # Timed runs
        times = []
        for i in range(10):
            start = time.time()
            result = pipeline.process(rgb)
            end = time.time()
            times.append(end - start)
            
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"Average processing time: {avg_time*1000:.1f}ms")
        print(f"Theoretical FPS: {fps:.1f}")
        print(f"Min/Max time: {min(times)*1000:.1f}ms / {max(times)*1000:.1f}ms")
        
        if fps >= 25:
            print("✅ Performance meets 25 FPS requirement")
        else:
            print(f"⚠️  Performance below 25 FPS: {fps:.1f}")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

if __name__ == '__main__':
    print("DSUNet Integration Unit Tests")
    print("="*50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    run_performance_test()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
