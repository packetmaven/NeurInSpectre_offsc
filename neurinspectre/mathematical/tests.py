#!/usr/bin/env python3
"""
NeurInSpectre Mathematical Foundations Test Module
Comprehensive testing suite for advanced mathematical capabilities
"""

import numpy as np
import sys
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MathematicalFoundationsTestSuite:
    """Comprehensive test suite for NeurInSpectre Mathematical Foundations"""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the test suite
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all mathematical foundations tests
        
        Returns:
            Dictionary with test results
        """
        print("🧪 NeurInSpectre Mathematical Foundations Test Suite")
        print("=" * 70)
        print()
        
        # Create sample data
        self.create_sample_data()
        
        # Run individual tests
        tests = [
            ("mathematical_foundations", self.test_mathematical_foundations),
            ("cli_integration", self.test_cli_integration),
            ("spectral_analysis", self.test_spectral_analysis),
            ("integration_schemes", self.test_integration_schemes),
            ("volterra_memory_analysis", self.test_volterra_memory_analysis),
            ("krylov_projection", self.test_krylov_projection),
            ("device_compatibility", self.test_device_compatibility),
            ("performance_benchmarks", self.test_performance_benchmarks)
        ]
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                self.test_results[test_name] = success
                if self.verbose:
                    print(f"✅ {test_name}: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = False
                if self.verbose:
                    import traceback
                    traceback.print_exc()
            print()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def test_mathematical_foundations(self) -> bool:
        """Test core mathematical foundations"""
        print("🧮 Testing NeurInSpectre Mathematical Foundations")
        print("=" * 60)
        
        try:
            # Import the mathematical foundations
            from . import GPUAcceleratedMathEngine, AdvancedExponentialIntegrator, demonstrate_advanced_mathematics
            
            print("✅ Successfully imported mathematical foundations!")
            print()
            
            # Test 1: Basic engine initialization
            print("🚀 Test 1: GPU Mathematical Engine Initialization")
            math_engine = GPUAcceleratedMathEngine(precision='float32', device_preference='auto')
            print(f"   Device: {math_engine.device}")
            print(f"   Precision: {math_engine.precision}")
            print()
            
            # Test 2: Spectral decomposition
            print("🔬 Test 2: Advanced Spectral Decomposition")
            test_gradient = np.random.randn(256) * 0.1
            results = math_engine.advanced_spectral_decomposition(test_gradient, decomposition_levels=3)
            
            print(f"   Spectral levels: {len(results['spectral_levels'])}")
            print(f"   Cross-correlations: {len(results['cross_correlations'])}")
            print(f"   Obfuscation indicators: {len(results['obfuscation_indicators'])}")
            
            if 'summary_metrics' in results and 'mean_entropy' in results['summary_metrics']:
                entropy = results['summary_metrics']['mean_entropy']
                print(f"   Mean entropy: {entropy.item():.4f}")
            print()
            
            # Test 3: Exponential integrator
            print("⚡ Test 3: Advanced Exponential Integrator")
            integrator = AdvancedExponentialIntegrator(math_engine)
            
            # Simple test function
            def test_nonlinear(u):
                import torch
                return -0.1 * u**2 + 0.05 * torch.sin(u)
            
            u_test = test_gradient[:50]  # Use smaller subset
            u_next = integrator.etd_rk4_step(u_test, None, test_nonlinear, 0.01)
            
            print(f"   Initial norm: {np.linalg.norm(u_test):.6f}")
            if hasattr(u_next, 'cpu'):
                final_norm = np.linalg.norm(u_next.cpu().numpy())
            else:
                final_norm = np.linalg.norm(u_next)
            print(f"   Final norm: {final_norm:.6f}")
            print()
            
            # Test 4: Full demonstration
            print("🎯 Test 4: Full Mathematical Demonstration")
            demonstrate_advanced_mathematics()
            print("   Demonstration completed successfully!")
            print()
            
            print("✅ All mathematical foundation tests passed!")
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Mathematical foundations may not be properly installed.")
            return False
        except Exception as e:
            print(f"❌ Test error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_cli_integration(self) -> bool:
        """Test CLI integration"""
        print("🖥️  Testing CLI Integration")
        print("=" * 60)
        
        try:
            # Test importing CLI components
            from ..cli.mathematical_commands import register_mathematical_commands
            print("✅ Successfully imported CLI mathematical commands!")
            
            # Test argument parser creation
            import argparse
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest='command')
            
            register_mathematical_commands(subparsers)
            print("✅ Successfully registered mathematical commands!")
            
            # Test help output
            help_output = parser.format_help()
            if 'math' in help_output:
                print("✅ Mathematical commands appear in help!")
            else:
                print("⚠️  Mathematical commands not found in help")

            # Ensure new paper-aligned subcommands exist under `math`
            math_parser = getattr(subparsers, "choices", {}).get("math")
            if math_parser is not None:
                math_help = math_parser.format_help()
                if "volterra" in math_help and "krylov" in math_help:
                    print("✅ Volterra + Krylov subcommands appear in `neurinspectre math --help`!")
                else:
                    print("⚠️  Volterra/Krylov missing from math help output")
            
            return True
            
        except ImportError as e:
            print(f"❌ CLI import error: {e}")
            return False
        except Exception as e:
            print(f"❌ CLI test error: {e}")
            return False

    def test_spectral_analysis(self) -> bool:
        """Test advanced spectral analysis capabilities"""
        print("🔬 Testing Advanced Spectral Analysis")
        print("=" * 60)
        
        try:
            from . import GPUAcceleratedMathEngine
            
            math_engine = GPUAcceleratedMathEngine(precision='float32', device_preference='auto')
            
            # Test different types of signals
            test_cases = [
                ("clean_signal", np.random.normal(0, 0.1, 512)),
                ("periodic_signal", np.sin(np.linspace(0, 4*np.pi, 512)) * 0.1),
                ("noisy_signal", np.random.normal(0, 0.1, 512) + 0.05 * np.sin(np.linspace(0, 8*np.pi, 512))),
                ("spike_signal", np.random.normal(0, 0.05, 512))
            ]
            
            # Add spikes to spike signal
            test_cases[3][1][::50] += 0.3 * np.random.randn(len(test_cases[3][1][::50]))
            
            for signal_name, signal_data in test_cases:
                print(f"   Testing {signal_name}...")
                results = math_engine.advanced_spectral_decomposition(signal_data, decomposition_levels=3)
                
                # Validate results structure
                assert 'spectral_levels' in results
                assert 'obfuscation_indicators' in results
                assert 'summary_metrics' in results
                
                entropy = results['summary_metrics']['mean_entropy'].item()
                print(f"     Mean entropy: {entropy:.4f}")
                
                if 'spectral_irregularity' in results['obfuscation_indicators']:
                    irregularity = results['obfuscation_indicators']['spectral_irregularity'].mean().item()
                    print(f"     Spectral irregularity: {irregularity:.4f}")
            
            print("✅ Spectral analysis tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Spectral analysis test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_integration_schemes(self) -> bool:
        """Test exponential time differencing integration schemes"""
        print("⚡ Testing Integration Schemes")
        print("=" * 60)
        
        try:
            from . import GPUAcceleratedMathEngine, AdvancedExponentialIntegrator
            import torch
            
            math_engine = GPUAcceleratedMathEngine(precision='float32', device_preference='auto')
            integrator = AdvancedExponentialIntegrator(math_engine)
            
            # Test different nonlinear functions
            test_functions = [
                ("linear", lambda u: -0.1 * u),
                ("quadratic", lambda u: -0.1 * u**2),
                ("cubic", lambda u: -0.1 * u**3),
                ("trigonometric", lambda u: 0.05 * torch.sin(u)),
                ("mixed", lambda u: -0.1 * u**2 + 0.05 * torch.sin(u))
            ]
            
            initial_state = np.random.randn(64) * 0.1
            
            for func_name, nonlinear_func in test_functions:
                print(f"   Testing {func_name} dynamics...")
                
                u_current = torch.tensor(initial_state, dtype=math_engine.precision, device=math_engine.device)
                
                # Perform multiple integration steps
                for step in range(5):
                    u_next = integrator.etd_rk4_step(u_current, None, nonlinear_func, 0.01)
                    
                    # Check for numerical stability
                    if torch.isnan(u_next).any() or torch.isinf(u_next).any():
                        raise ValueError(f"Numerical instability in {func_name} at step {step}")
                    
                    u_current = u_next
                
                final_norm = torch.norm(u_current).item()
                print(f"     Final norm after 5 steps: {final_norm:.6f}")
            
            print("✅ Integration scheme tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Integration scheme test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_volterra_memory_analysis(self) -> bool:
        """Test Volterra (Layer 2) power-law kernel fitting"""
        print("🧠 Testing Volterra Memory Analysis")
        print("=" * 60)
        try:
            from . import fit_volterra_power_law, predict_volterra_power_law

            rng = np.random.default_rng(123)
            n = 80
            dt = 0.01
            alpha_true = 0.60
            c_true = 2.00
            y0 = 1.0

            y_clean = predict_volterra_power_law(y0, alpha=alpha_true, c=c_true, n=n, dt=dt)
            y_obs = y_clean + 0.01 * rng.normal(size=n)

            res = fit_volterra_power_law(
                y_obs,
                dt=dt,
                normalize="by_y0",
                alpha_bounds=(0.05, 0.995),
                c_bounds=(1e-6, 10.0),
                maxiter=120,
                seed=42,
            )

            print(f"   alpha_hat={res.alpha:.3f}, c_hat={res.c:.3f}, rmse={res.rmse_scaled:.4f} (scaled)")
            assert 0.05 <= float(res.alpha) <= 0.995
            assert float(res.c) > 0.0
            assert np.isfinite(res.rmse) and np.isfinite(res.rmse_scaled)

            print("✅ Volterra memory analysis tests passed!")
            return True
        except Exception as e:
            print(f"❌ Volterra memory analysis test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_krylov_projection(self) -> bool:
        """Test Krylov projection diagnostics (Layer 3)"""
        print("🧩 Testing Krylov Projection")
        print("=" * 60)
        try:
            from .krylov import laplacian_1d_matvec, analyze_krylov_projection
            from scipy.linalg import expm

            rng = np.random.default_rng(7)
            T = 8
            D = 30  # small enough to validate with dense expm
            dt = 0.05
            damping = 0.1

            # Build dense L for reference generation
            L = np.zeros((D, D), dtype=np.float64)
            for i in range(D):
                L[i, i] = -2.0 - damping
                if i - 1 >= 0:
                    L[i, i - 1] = 1.0
                if i + 1 < D:
                    L[i, i + 1] = 1.0
            A = expm(dt * L)

            seq = np.zeros((T, D), dtype=np.float64)
            seq[0] = rng.normal(0, 1, size=D)
            for t in range(T - 1):
                seq[t + 1] = A @ seq[t]

            summary, per_step, eigvals0 = analyze_krylov_projection(
                seq,
                krylov_dim=D,  # full basis => near-exact
                dt=dt,
                damping=damping,
                steps=T - 1,
                stride=1,
                atol=1e-14,
            )

            mean_err = float(summary["reconstruction_error"]["mean"])
            print(f"   mean_rel_err={mean_err:.6f}, dissipation_anomaly={summary['dissipation']['dissipation_anomaly_score']:.3f}")
            assert np.isfinite(mean_err)
            assert mean_err < 1e-3
            assert summary["dissipation"]["dissipation_anomaly_score"] <= 0.25
            assert isinstance(per_step, list) and len(per_step) >= 1
            assert eigvals0 is not None

            # Basic sanity for matvec
            x = rng.normal(0, 1, size=D)
            y = laplacian_1d_matvec(x, damping=damping)
            assert y.shape == x.shape

            print("✅ Krylov projection tests passed!")
            return True
        except Exception as e:
            print(f"❌ Krylov projection test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_device_compatibility(self) -> bool:
        """Test compatibility across different devices"""
        print("🖥️  Testing Device Compatibility")
        print("=" * 60)
        
        try:
            from . import GPUAcceleratedMathEngine
            import torch
            
            # Test all available devices
            devices_to_test = ['cpu']
            
            if torch.cuda.is_available():
                devices_to_test.append('cuda')
                print("   CUDA device available")
            
            if torch.backends.mps.is_available():
                devices_to_test.append('mps')
                print("   MPS device available")
            
            test_data = np.random.randn(128) * 0.1
            
            for device in devices_to_test:
                print(f"   Testing on {device.upper()}...")
                
                try:
                    math_engine = GPUAcceleratedMathEngine(
                        precision='float32', 
                        device_preference=device
                    )
                    
                    # Test spectral analysis
                    results = math_engine.advanced_spectral_decomposition(test_data, decomposition_levels=2)
                    
                    # Verify results are on correct device
                    for level_data in results['spectral_levels'].values():
                        if hasattr(level_data['magnitude'], 'device'):
                            assert str(level_data['magnitude'].device).startswith(device) or device == 'cpu'
                    
                    print(f"     ✅ {device.upper()} test passed")
                    
                except Exception as e:
                    print(f"     ⚠️  {device.upper()} test failed: {e}")
                    if device == 'cpu':  # CPU should always work
                        raise
            
            print("✅ Device compatibility tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Device compatibility test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks"""
        print("⚡ Testing Performance Benchmarks")
        print("=" * 60)
        
        try:
            from . import GPUAcceleratedMathEngine
            import time
            
            math_engine = GPUAcceleratedMathEngine(precision='float32', device_preference='auto')
            
            # Test different data sizes
            data_sizes = [128, 256, 512, 1024]
            
            for size in data_sizes:
                test_data = np.random.randn(size) * 0.1
                
                # Benchmark spectral analysis
                start_time = time.time()
                results = math_engine.advanced_spectral_decomposition(test_data, decomposition_levels=3)
                end_time = time.time()
                
                processing_time = end_time - start_time
                print(f"   Size {size}: {processing_time:.4f}s ({size/processing_time:.0f} samples/sec)")
                
                # Verify results quality
                assert 'spectral_levels' in results
                assert len(results['spectral_levels']) >= 3
            
            print("✅ Performance benchmark tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Performance benchmark test failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def create_sample_data(self):
        """Create sample gradient data for testing"""
        print("📊 Creating sample gradient data for testing...")
        
        # Create realistic gradient data with different patterns
        n_samples = 512
        
        # Clean gradients (normal distribution)
        clean_gradients = np.random.normal(0, 0.1, n_samples)
        
        # Obfuscated gradients (with artificial patterns)
        obfuscated_gradients = clean_gradients.copy()
        obfuscated_gradients += 0.05 * np.sin(np.arange(n_samples) * 0.1)  # Periodic component
        obfuscated_gradients[::50] += 0.3 * np.random.randn(len(obfuscated_gradients[::50]))  # Spikes
        
        # Save sample data
        np.save('sample_clean_gradients.npy', clean_gradients)
        np.save('sample_obfuscated_gradients.npy', obfuscated_gradients)
        
        print("✅ Sample data created:")
        print("   • sample_clean_gradients.npy")
        print("   • sample_obfuscated_gradients.npy")
        print()

    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("📋 Test Results Summary")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<30}: {status}")
        
        print()
        print(f"Overall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Mathematical foundations are ready to use.")
            print()
            print("🚀 Try these commands:")
            print("   python -m neurinspectre.cli math demo")
            print("   python -m neurinspectre.cli math spectral --input sample_clean_gradients.npy")
            print("   python -m neurinspectre.cli math integrate --input sample_obfuscated_gradients.npy --steps 50")
        else:
            print("❌ Some tests failed. Please check the installation.")
        
        print()

def run_test_suite(verbose: bool = False) -> bool:
    """
    Run the complete mathematical foundations test suite
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        True if all tests pass, False otherwise
    """
    test_suite = MathematicalFoundationsTestSuite(verbose=verbose)
    results = test_suite.run_all_tests()
    return all(results.values())

def main():
    """Main test function for standalone execution"""
    import argparse

    # Configure logging for standalone execution (avoid import-time side effects).
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="NeurInSpectre Mathematical Foundations Test Suite")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    success = run_test_suite(verbose=args.verbose)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 