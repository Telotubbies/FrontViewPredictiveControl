#!/usr/bin/env python3
"""
Performance Benchmark for CARLA MPC Pipeline
Compare original vs optimized implementations
"""

import sys
import time
import numpy as np
import torch
import logging
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import CAM_W, CAM_H, BEV_W, BEV_H
from perception.lane_trajectory import bev_transform, sliding_window_search
from perception.gpu_bev_transform import bev_transform_gpu, get_bev_transform_matrix_gpu
from perception.vectorized_sliding_window import sliding_window_search_gpu
from alg.optimized_reference import generate_reference_path_optimized, evaluate_reference_at_horizon

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark performance improvements"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def create_test_data(self):
        """Create test data for benchmarking"""
        # Test mask
        mask = np.zeros((CAM_H, CAM_W), dtype=np.uint8)
        mask[100:380, 200:440] = 255  # Lane region
        
        # Test BEV mask
        bev_mask = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
        bev_mask[50:270, 150:200] = 255  # Left lane
        bev_mask[50:270, 250:300] = 255  # Right lane
        
        # Test coefficients
        left_coeffs = np.array([0.001, 0.0, -1.5])
        right_coeffs = np.array([0.001, 0.0, 1.5])
        
        return mask, bev_mask, left_coeffs, right_coeffs
    
    def benchmark_bev_transform(self, mask: np.ndarray, n_runs: int = 100):
        """Benchmark BEV transform CPU vs GPU"""
        logger.info("Benchmarking BEV transform...")
        
        # CPU version
        start_time = time.time()
        for _ in range(n_runs):
            bev_cpu = bev_transform(mask, CAM_W, CAM_H)
        cpu_time = (time.time() - start_time) / n_runs
        
        # GPU version
        if torch.cuda.is_available():
            mask_tensor = torch.from_numpy(mask).float().to(self.device)
            M_gpu = get_bev_transform_matrix_gpu(CAM_W, CAM_H, device=self.device)
            
            start_time = time.time()
            for _ in range(n_runs):
                bev_gpu = bev_transform_gpu(mask_tensor, M_gpu, (BEV_W, BEV_H))
                torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / n_runs
            
            speedup = cpu_time / gpu_time
            self.results['bev_transform'] = {
                'cpu_ms': cpu_time * 1000,
                'gpu_ms': gpu_time * 1000,
                'speedup': speedup
            }
            
            logger.info(f"BEV Transform: CPU {cpu_time*1000:.2f}ms, GPU {gpu_time*1000:.2f}ms, Speedup {speedup:.2f}x")
        else:
            self.results['bev_transform'] = {
                'cpu_ms': cpu_time * 1000,
                'gpu_ms': None,
                'speedup': None
            }
            logger.info(f"BEV Transform: CPU {cpu_time*1000:.2f}ms (GPU not available)")
    
    def benchmark_sliding_window(self, bev_mask: np.ndarray, n_runs: int = 50):
        """Benchmark sliding window CPU vs GPU"""
        logger.info("Benchmarking sliding window...")
        
        # CPU version
        start_time = time.time()
        for _ in range(n_runs):
            result_cpu = sliding_window_search(bev_mask)
        cpu_time = (time.time() - start_time) / n_runs
        
        # GPU version
        if torch.cuda.is_available():
            bev_tensor = torch.from_numpy(bev_mask).float().to(self.device)
            
            start_time = time.time()
            for _ in range(n_runs):
                result_gpu = sliding_window_search_gpu(bev_tensor)
                torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / n_runs
            
            speedup = cpu_time / gpu_time
            self.results['sliding_window'] = {
                'cpu_ms': cpu_time * 1000,
                'gpu_ms': gpu_time * 1000,
                'speedup': speedup
            }
            
            logger.info(f"Sliding Window: CPU {cpu_time*1000:.2f}ms, GPU {gpu_time*1000:.2f}ms, Speedup {speedup:.2f}x")
        else:
            self.results['sliding_window'] = {
                'cpu_ms': cpu_time * 1000,
                'gpu_ms': None,
                'speedup': None
            }
            logger.info(f"Sliding Window: CPU {cpu_time*1000:.2f}ms (GPU not available)")
    
    def benchmark_reference_generation(self, left_coeffs: np.ndarray, right_coeffs: np.ndarray, n_runs: int = 1000):
        """Benchmark reference path generation"""
        logger.info("Benchmarking reference path generation...")
        
        # Original method (simulate)
        x_sample = np.linspace(0, 30, 50)
        
        start_time = time.time()
        for _ in range(n_runs):
            # Simulate original method: sample + polyval
            left_y = np.polyval(left_coeffs, x_sample)
            right_y = np.polyval(right_coeffs, x_sample)
            center_y = (left_y + right_y) / 2
            dy_dx = np.gradient(center_y, x_sample)
            yaw = np.arctan2(dy_dx, 1)
        original_time = (time.time() - start_time) / n_runs
        
        # Optimized method
        start_time = time.time()
        for _ in range(n_runs):
            x_ref, y_ref, yaw_ref, coeffs = generate_reference_path_optimized(
                waypoints=None,
                left_coeffs=left_coeffs,
                right_coeffs=right_coeffs
            )
        optimized_time = (time.time() - start_time) / n_runs
        
        speedup = original_time / optimized_time
        self.results['reference_generation'] = {
            'cpu_ms': original_time * 1000,
            'gpu_ms': optimized_time * 1000,
            'speedup': speedup
        }      
        logger.info(f"Reference Generation: Original {original_time*1000:.2f}ms, Optimized {optimized_time*1000:.2f}ms, Speedup {speedup:.2f}x")
    
    def benchmark_mpc_evaluation(self, coeffs: np.ndarray, n_runs: int = 1000):
        """Benchmark MPC horizon evaluation"""
        logger.info("Benchmarking MPC evaluation...")
        
        # Original method (simulate redundant sampling)
        mpc_horizon = np.arange(1, 11) * 2.0
        
        start_time = time.time()
        for _ in range(n_runs):
            # Simulate original: sample at many points, then interpolate
            x_dense = np.linspace(0, 30, 200)
            y_dense = np.polyval(coeffs, x_dense)
            # Interpolate to MPC points
            y_mpc = np.interp(mpc_horizon, x_dense, y_dense)
        original_time = (time.time() - start_time) / n_runs
        
        # Optimized method: direct evaluation
        start_time = time.time()
        for _ in range(n_runs):
            y_mpc = evaluate_reference_at_horizon(coeffs, mpc_horizon)
        optimized_time = (time.time() - start_time) / n_runs
        
        speedup = original_time / optimized_time
        self.results['mpc_evaluation'] = {
            'cpu_ms': original_time * 1000,
            'gpu_ms': optimized_time * 1000,
            'speedup': speedup
        }
        
        logger.info(f"MPC Evaluation: Original {original_time*1000:.2f}ms, Optimized {optimized_time*1000:.2f}ms, Speedup {speedup:.2f}x")
    
    def run_full_benchmark(self):
        """Run complete performance benchmark"""
        logger.info("Starting full performance benchmark...")
        logger.info(f"Using device: {self.device}")
        
        # Create test data
        mask, bev_mask, left_coeffs, right_coeffs = self.create_test_data()
        
        # Run benchmarks
        self.benchmark_bev_transform(mask)
        self.benchmark_sliding_window(bev_mask)
        self.benchmark_reference_generation(left_coeffs, right_coeffs)
        self.benchmark_mpc_evaluation((left_coeffs + right_coeffs) / 2)
        
        # Calculate overall improvement
        self.calculate_total_improvement()
        
        # Print summary
        self.print_summary()
    
    def calculate_total_improvement(self):
        """Calculate total pipeline improvement"""
        total_cpu_time = 0
        total_gpu_time = 0
        
        for component, result in self.results.items():
            if 'cpu_ms' in result and result['cpu_ms'] is not None:
                total_cpu_time += result['cpu_ms']
                if 'gpu_ms' in result and result['gpu_ms'] is not None:
                    total_gpu_time += result['gpu_ms']
                else:
                    total_gpu_time += result['cpu_ms']  # No GPU improvement
        
        if total_gpu_time > 0:
            overall_speedup = total_cpu_time / total_gpu_time
            self.results['overall'] = {
                'total_cpu_ms': total_cpu_time,
                'total_gpu_ms': total_gpu_time,
                'overall_speedup': overall_speedup
            }
    
    def print_summary(self):
        """Print benchmark summary"""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE BENCHMARK SUMMARY")
        logger.info("="*60)
        
        for component, result in self.results.items():
            if component == 'overall':
                logger.info(f"\nOVERALL PIPELINE:")
                logger.info(f"  Total CPU time: {result['total_cpu_ms']:.2f}ms")
                logger.info(f"  Total GPU time: {result['total_gpu_ms']:.2f}ms")
                logger.info(f"  Overall speedup: {result['overall_speedup']:.2f}x")
            else:
                logger.info(f"\n{component.upper().replace('_', ' ')}:")
                logger.info(f"  CPU: {result['cpu_ms']:.2f}ms")
                if result.get('gpu_ms') is not None:
                    logger.info(f"  GPU: {result['gpu_ms']:.2f}ms")
                    logger.info(f"  Speedup: {result['speedup']:.2f}x")
                else:
                    logger.info("  GPU: Not available")
        
        logger.info("\n" + "="*60)


def main():
    """Main benchmark runner"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    benchmark = PerformanceBenchmark()
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
