#!/usr/bin/env python3
# run_tests.py - Test runner script for ML Project Effort Estimator
"""
Comprehensive test runner for the ML Project Effort Estimator
Provides different test execution modes for development and production validation
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully ({duration:.1f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} failed ({duration:.1f}s)")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False
    
    return True

def check_prerequisites():
    """Check that prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check pytest is installed
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} available")
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest")
        return False
    
    # Check if models folder exists
    if Path("models").exists():
        model_files = list(Path("models").glob("*.pkl"))
        print(f"✅ Models folder found with {len(model_files)} .pkl files")
    else:
        print("⚠️ Models folder not found - some tests will be skipped")
    
    # Check if config folder exists
    if Path("config").exists():
        config_files = list(Path("config").glob("*.yaml"))
        print(f"✅ Config folder found with {len(config_files)} .yaml files")
    else:
        print("⚠️ Config folder not found - some tests may fail")
    
    return True

def run_fast_tests():
    """Run fast unit tests for development feedback"""
    cmd = "python -m pytest tests/unit/ -v --tb=short -x"
    return run_command(cmd, "Fast Unit Tests")

def run_integration_tests():
    """Run integration tests"""
    cmd = "python -m pytest tests/integration/ -v --tb=short"
    return run_command(cmd, "Integration Tests")

def run_e2e_tests():
    """Run end-to-end tests"""
    cmd = "python -m pytest tests/e2e/ -v --tb=short -s"
    return run_command(cmd, "End-to-End Tests")

def run_benchmarks():
    """Run performance benchmarks"""
    cmd = "python -m pytest tests/benchmarks/ -v --tb=short -s"
    return run_command(cmd, "Performance Benchmarks")

def run_production_validation():
    """Run production readiness validation"""
    cmd = "python -m pytest tests/e2e/test_production_workflow.py::TestProductionReadiness -v -s"
    return run_command(cmd, "Production Readiness Validation")

def run_all_tests():
    """Run complete test suite"""
    cmd = "python -m pytest tests/ -v --tb=short"
    return run_command(cmd, "Complete Test Suite")

def run_coverage_report():
    """Run tests with coverage report"""
    cmd = "python -m pytest tests/ --cov=. --cov-report=html --cov-report=term"
    return run_command(cmd, "Test Coverage Analysis")

def run_stress_tests():
    """Run stress tests (marked as slow)"""
    cmd = "python -m pytest tests/ -v -m slow --tb=short"
    return run_command(cmd, "Stress Tests")

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="ML Project Effort Estimator Test Runner")
    
    parser.add_argument("--fast", action="store_true", 
                       help="Run fast unit tests only (for development)")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests")
    parser.add_argument("--e2e", action="store_true",
                       help="Run end-to-end tests")
    parser.add_argument("--benchmarks", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--production", action="store_true",
                       help="Run production readiness validation")
    parser.add_argument("--all", action="store_true",
                       help="Run complete test suite")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage report")
    parser.add_argument("--stress", action="store_true",
                       help="Run stress tests")
    parser.add_argument("--check", action="store_true",
                       help="Check prerequisites only")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed")
        sys.exit(1)
    
    if args.check:
        print("\n✅ Prerequisites check completed")
        sys.exit(0)
    
    # Determine which tests to run
    tests_to_run = []
    
    if args.fast:
        tests_to_run.append(("Fast Tests", run_fast_tests))
    elif args.integration:
        tests_to_run.append(("Integration Tests", run_integration_tests))
    elif args.e2e:
        tests_to_run.append(("End-to-End Tests", run_e2e_tests))
    elif args.benchmarks:
        tests_to_run.append(("Performance Benchmarks", run_benchmarks))
    elif args.production:
        tests_to_run.append(("Production Validation", run_production_validation))
    elif args.coverage:
        tests_to_run.append(("Coverage Report", run_coverage_report))
    elif args.stress:
        tests_to_run.append(("Stress Tests", run_stress_tests))
    elif args.all:
        tests_to_run = [
            ("Fast Tests", run_fast_tests),
            ("Integration Tests", run_integration_tests),
            ("End-to-End Tests", run_e2e_tests),
            ("Performance Benchmarks", run_benchmarks),
            ("Production Validation", run_production_validation)
        ]
    else:
        # Default: run fast tests
        tests_to_run.append(("Fast Tests (default)", run_fast_tests))
    
    # Run selected tests
    start_time = time.time()
    passed_tests = 0
    failed_tests = 0
    
    for test_name, test_func in tests_to_run:
        if test_func():
            passed_tests += 1
        else:
            failed_tests += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.1f}s")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 {failed_tests} test suite(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()