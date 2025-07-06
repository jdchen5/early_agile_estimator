# benchmarks/test_performance_benchmarks.py - Performance monitoring and benchmarks
"""
Performance benchmarks for monitoring system performance over time
Stores results in benchmarks/results/ for trend analysis
"""

import pytest
import time
import json
import psutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Import components to benchmark
from models import (
    list_available_models,
    predict_man_hours,
    prepare_features_for_model,
    load_model
)

class PerformanceBenchmark:
    """Performance benchmark runner and results storage"""
    
    def __init__(self):
        self.results_dir = Path("benchmarks/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {},
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'process_id': os.getpid()
        }
    
    def run_benchmark(self, name: str, func, *args, **kwargs):
        """Run a benchmark function and record results"""
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Record results
        benchmark_result = {
            'duration_seconds': end_time - start_time,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_delta_mb': memory_after - memory_before,
            'success': success,
            'error': error,
            'result_size': len(str(result)) if result is not None else 0
        }
        
        self.session_results['benchmarks'][name] = benchmark_result
        return result, benchmark_result
    
    def save_results(self):
        """Save benchmark results to file"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        
        # Create date directory
        date_dir = self.results_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = date_dir / f"benchmark_{time_str}.json"
        with open(results_file, 'w') as f:
            json.dump(self.session_results, f, indent=2)
        
        # Update summary
        self._update_summary(date_dir)
        
        return results_file
    
    def _update_summary(self, date_dir: Path):
        """Update daily summary with latest results"""
        summary_file = date_dir / "daily_summary.json"
        
        # Load existing summary or create new
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'date': date_dir.name,
                'benchmark_runs': [],
                'performance_trends': {}
            }
        
        # Add current run
        run_summary = {
            'timestamp': self.session_results['timestamp'],
            'benchmarks': {name: {
                'duration': result['duration_seconds'],
                'memory_delta': result['memory_delta_mb'],
                'success': result['success']
            } for name, result in self.session_results['benchmarks'].items()}
        }
        
        summary['benchmark_runs'].append(run_summary)
        
        # Calculate trends
        self._calculate_trends(summary)
        
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _calculate_trends(self, summary: Dict):
        """Calculate performance trends from multiple runs"""
        if len(summary['benchmark_runs']) < 2:
            return
        
        # Group by benchmark name
        benchmark_data = {}
        for run in summary['benchmark_runs']:
            for bench_name, bench_data in run['benchmarks'].items():
                if bench_name not in benchmark_data:
                    benchmark_data[bench_name] = []
                benchmark_data[bench_name].append(bench_data)
        
        # Calculate trends
        trends = {}
        for bench_name, data_points in benchmark_data.items():
            if len(data_points) >= 2:
                durations = [d['duration'] for d in data_points if d['success']]
                if durations:
                    trends[bench_name] = {
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations),
                        'recent_duration': durations[-1],
                        'trend': 'improving' if durations[-1] < durations[0] else 'degrading',
                        'sample_count': len(durations)
                    }
        
        summary['performance_trends'] = trends

@pytest.fixture
def benchmark_runner():
    """Performance benchmark runner fixture"""
    return PerformanceBenchmark()

class TestPredictionPerformance:
    """Benchmark prediction performance"""
    
    def test_single_prediction_benchmark(self, benchmark_runner, sample_ui_inputs, benchmark_config):
        """Benchmark single prediction performance"""
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models available for benchmarking")
        
        model_name = available_models[0]['technical_name']
        
        # Benchmark prediction
        result, bench_result = benchmark_runner.run_benchmark(
            'single_prediction',
            predict_man_hours,
            sample_ui_inputs,
            model_name
        )
        
        # Validate performance
        target_time = benchmark_config['targets']['prediction_time_seconds']
        actual_time = bench_result['duration_seconds']
        
        assert bench_result['success'], f"Prediction failed: {bench_result['error']}"
        
        if actual_time > target_time:
            print(f"⚠️ Performance regression: {actual_time:.2f}s > {target_time}s")
        else:
            print(f"✅ Prediction performance: {actual_time:.2f}s (target: {target_time}s)")
        
        # Memory check
        target_memory = benchmark_config['targets']['memory_usage_mb']
        if bench_result['memory_after_mb'] > target_memory:
            print(f"⚠️ Memory usage high: {bench_result['memory_after_mb']:.1f}MB")
    
    def test_multi_model_prediction_benchmark(self, benchmark_runner, sample_ui_inputs, benchmark_config):
        """Benchmark multi-model prediction performance"""
        available_models = list_available_models()
        
        if len(available_models) < 2:
            pytest.skip("Need multiple models for multi-model benchmark")
        
        # Test up to 3 models
        test_models = available_models[:min(3, len(available_models))]
        
        def multi_model_prediction():
            results = {}
            for model_info in test_models:
                model_name = model_info['technical_name']
                prediction = predict_man_hours(sample_ui_inputs, model_name)
                if prediction is not None:
                    results[model_name] = prediction
            return results
        
        result, bench_result = benchmark_runner.run_benchmark(
            'multi_model_prediction',
            multi_model_prediction
        )
        
        # Validate performance
        target_time = benchmark_config['targets']['multi_model_time_seconds']
        actual_time = bench_result['duration_seconds']
        
        assert bench_result['success'], f"Multi-model prediction failed: {bench_result['error']}"
        
        if actual_time > target_time:
            print(f"⚠️ Multi-model performance regression: {actual_time:.2f}s > {target_time}s")
        else:
            print(f"✅ Multi-model performance: {actual_time:.2f}s (target: {target_time}s)")
        
        print(f"✅ Models tested: {len(result) if result else 0}")

class TestFeaturePreparationPerformance:
    """Benchmark feature preparation performance"""
    
    def test_feature_preparation_benchmark(self, benchmark_runner, sample_ui_inputs):
        """Benchmark feature preparation performance"""
        
        result, bench_result = benchmark_runner.run_benchmark(
            'feature_preparation',
            prepare_features_for_model,
            sample_ui_inputs
        )
        
        assert bench_result['success'], f"Feature preparation failed: {bench_result['error']}"
        
        # Feature preparation should be fast
        if bench_result['duration_seconds'] > 2.0:
            print(f"⚠️ Feature preparation slow: {bench_result['duration_seconds']:.2f}s")
        else:
            print(f"✅ Feature preparation: {bench_result['duration_seconds']:.2f}s")
        
        if result is not None:
            print(f"✅ Features generated: {result.shape}")
    
    def test_batch_feature_preparation_benchmark(self, benchmark_runner, sample_ui_inputs):
        """Benchmark batch feature preparation"""
        
        def batch_feature_preparation():
            batch_size = 5
            results = []
            
            for i in range(batch_size):
                # Vary inputs slightly for each batch item
                modified_inputs = sample_ui_inputs.copy()
                modified_inputs['project_prf_functional_size'] = \
                    sample_ui_inputs['project_prf_functional_size'] + (i * 10)
                
                result = prepare_features_for_model(modified_inputs)
                if result is not None:
                    results.append(result)
            
            return results
        
        result, bench_result = benchmark_runner.run_benchmark(
            'batch_feature_preparation',
            batch_feature_preparation
        )
        
        assert bench_result['success'], f"Batch feature preparation failed: {bench_result['error']}"
        
        if result:
            avg_time_per_item = bench_result['duration_seconds'] / len(result)
            print(f"✅ Batch feature preparation: {avg_time_per_item:.3f}s per item")

class TestModelLoadingPerformance:
    """Benchmark model loading performance"""
    
    def test_model_loading_benchmark(self, benchmark_runner):
        """Benchmark model loading (cold start)"""
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models available for loading benchmark")
        
        model_name = available_models[0]['technical_name']
        
        result, bench_result = benchmark_runner.run_benchmark(
            'model_loading',
            load_model,
            model_name
        )
        
        assert bench_result['success'], f"Model loading failed: {bench_result['error']}"
        
        # Model loading should complete within reasonable time
        if bench_result['duration_seconds'] > 10.0:
            print(f"⚠️ Model loading slow: {bench_result['duration_seconds']:.2f}s")
        else:
            print(f"✅ Model loading: {bench_result['duration_seconds']:.2f}s")
    
    def test_multiple_model_loading_benchmark(self, benchmark_runner):
        """Benchmark loading multiple models"""
        available_models = list_available_models()
        
        if len(available_models) < 2:
            pytest.skip("Need multiple models for batch loading benchmark")
        
        def load_multiple_models():
            loaded_models = {}
            for model_info in available_models[:3]:  # Load up to 3 models
                model_name = model_info['technical_name']
                model = load_model(model_name)
                if model is not None:
                    loaded_models[model_name] = model
            return loaded_models
        
        result, bench_result = benchmark_runner.run_benchmark(
            'multiple_model_loading',
            load_multiple_models
        )
        
        assert bench_result['success'], f"Multiple model loading failed: {bench_result['error']}"
        
        if result:
            avg_time_per_model = bench_result['duration_seconds'] / len(result)
            print(f"✅ Multiple model loading: {avg_time_per_model:.2f}s per model")
            print(f"✅ Total memory increase: {bench_result['memory_delta_mb']:.1f}MB")

class TestMemoryUsageBenchmarks:
    """Benchmark memory usage patterns"""
    
    def test_memory_usage_single_prediction(self, benchmark_runner, sample_ui_inputs, benchmark_config):
        """Monitor memory usage during single prediction"""
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models available for memory benchmarking")
        
        model_name = available_models[0]['technical_name']
        
        # Multiple predictions to check for memory leaks
        def multiple_predictions():
            results = []
            for i in range(5):
                prediction = predict_man_hours(sample_ui_inputs, model_name)
                if prediction is not None:
                    results.append(prediction)
            return results
        
        result, bench_result = benchmark_runner.run_benchmark(
            'memory_usage_multiple_predictions',
            multiple_predictions
        )
        
        assert bench_result['success'], f"Memory benchmark failed: {bench_result['error']}"
        
        # Check memory growth
        memory_per_prediction = bench_result['memory_delta_mb'] / len(result) if result else 0
        
        if memory_per_prediction > 10:  # More than 10MB per prediction suggests leak
            print(f"⚠️ Potential memory leak: {memory_per_prediction:.2f}MB per prediction")
        else:
            print(f"✅ Memory usage stable: {memory_per_prediction:.2f}MB per prediction")
        
        # Total memory check
        target_memory = benchmark_config['targets']['memory_usage_mb']
        if bench_result['memory_after_mb'] > target_memory:
            print(f"⚠️ Memory usage exceeds target: {bench_result['memory_after_mb']:.1f}MB > {target_memory}MB")

class TestConcurrencyBenchmarks:
    """Benchmark concurrent operations"""
    
    def test_concurrent_predictions_benchmark(self, benchmark_runner, sample_ui_inputs):
        """Benchmark concurrent prediction handling"""
        import threading
        import queue
        
        available_models = list_available_models()
        
        if not available_models:
            pytest.skip("No models available for concurrency benchmarking")
        
        model_name = available_models[0]['technical_name']
        
        def concurrent_predictions():
            num_threads = 3
            results_queue = queue.Queue()
            
            def worker():
                prediction = predict_man_hours(sample_ui_inputs, model_name)
                results_queue.put(prediction)
            
            # Start threads
            threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            return results
        
        result, bench_result = benchmark_runner.run_benchmark(
            'concurrent_predictions',
            concurrent_predictions
        )
        
        assert bench_result['success'], f"Concurrent prediction benchmark failed: {bench_result['error']}"
        
        if result:
            success_rate = len([r for r in result if r is not None]) / len(result)
            print(f"✅ Concurrent predictions success rate: {success_rate:.1%}")
            print(f"✅ Concurrent execution time: {bench_result['duration_seconds']:.2f}s")

class TestRegressionDetection:
    """Detect performance regressions by comparing with historical data"""
    
    def test_performance_regression_detection(self, benchmark_runner, benchmark_config):
        """Check for performance regressions against historical baselines"""
        
        # Try to load baseline data
        baseline_file = Path("benchmarks/baselines/baseline_metrics.json")
        
        if not baseline_file.exists():
            print("⚠️ No baseline metrics found - this run will establish baseline")
            return
        
        with open(baseline_file, 'r') as f:
            baseline_metrics = json.load(f)
        
        current_results = benchmark_runner.session_results['benchmarks']
        tolerance_percent = benchmark_config['tolerance']['time_regression_percent']
        
        regressions_detected = []
        
        for benchmark_name, current_result in current_results.items():
            if benchmark_name in baseline_metrics and current_result['success']:
                baseline_time = baseline_metrics[benchmark_name]['duration_seconds']
                current_time = current_result['duration_seconds']
                
                # Calculate regression percentage
                regression_percent = ((current_time - baseline_time) / baseline_time) * 100
                
                if regression_percent > tolerance_percent:
                    regressions_detected.append({
                        'benchmark': benchmark_name,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'regression_percent': regression_percent
                    })
        
        # Report regressions
        if regressions_detected:
            print("⚠️ Performance regressions detected:")
            for regression in regressions_detected:
                print(f"   {regression['benchmark']}: "
                     f"{regression['current_time']:.2f}s vs {regression['baseline_time']:.2f}s "
                     f"({regression['regression_percent']:+.1f}%)")
        else:
            print("✅ No performance regressions detected")
    
    def test_update_baseline_metrics(self, benchmark_runner):
        """Update baseline metrics with current results"""
        
        baseline_dir = Path("benchmarks/baselines")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_file = baseline_dir / "baseline_metrics.json"
        
        # Extract key metrics from current run
        baseline_metrics = {}
        for benchmark_name, result in benchmark_runner.session_results['benchmarks'].items():
            if result['success']:
                baseline_metrics[benchmark_name] = {
                    'duration_seconds': result['duration_seconds'],
                    'memory_delta_mb': result['memory_delta_mb'],
                    'timestamp': benchmark_runner.session_results['timestamp']
                }
        
        # Save baseline
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
        
        print(f"✅ Baseline metrics updated: {len(baseline_metrics)} benchmarks")

@pytest.fixture(autouse=True)
def save_benchmark_results(benchmark_runner):
    """Automatically save benchmark results after test session"""
    yield
    
    # Save results after all benchmarks complete
    if benchmark_runner.session_results['benchmarks']:
        results_file = benchmark_runner.save_results()
        print(f"\n✅ Benchmark results saved: {results_file}")

class TestBenchmarkReporting:
    """Generate benchmark reports and summaries"""
    
    def test_generate_benchmark_summary(self, benchmark_runner):
        """Generate human-readable benchmark summary"""
        
        if not benchmark_runner.session_results['benchmarks']:
            pytest.skip("No benchmark results to summarize")
        
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Timestamp: {benchmark_runner.session_results['timestamp']}")
        print(f"System: {benchmark_runner.session_results['system_info']['cpu_count']} CPU cores, "
              f"{benchmark_runner.session_results['system_info']['memory_total_gb']:.1f}GB RAM")
        
        print("\nBenchmark Results:")
        print("-" * 40)
        
        for benchmark_name, result in benchmark_runner.session_results['benchmarks'].items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration_seconds']
            memory = result['memory_delta_mb']
            
            print(f"{benchmark_name:30} {status}")
            print(f"{'':30} Time: {duration:.3f}s, Memory: {memory:+.1f}MB")
        
        print("\n" + "="*60)
    
    def test_performance_trend_analysis(self):
        """Analyze performance trends from historical data"""
        
        results_dir = Path("benchmarks/results")
        
        if not results_dir.exists():
            pytest.skip("No historical benchmark data available")
        
        # Find recent summary files
        summary_files = []
        for date_dir in results_dir.iterdir():
            if date_dir.is_dir():
                summary_file = date_dir / "daily_summary.json"
                if summary_file.exists():
                    summary_files.append(summary_file)
        
        if len(summary_files) < 2:
            pytest.skip("Need at least 2 days of data for trend analysis")
        
        print("\n" + "="*60)
        print("PERFORMANCE TREND ANALYSIS")
        print("="*60)
        
        # Load and analyze trends
        for summary_file in sorted(summary_files)[-5:]:  # Last 5 days
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print(f"\nDate: {summary['date']}")
            
            if 'performance_trends' in summary:
                for benchmark_name, trend_data in summary['performance_trends'].items():
                    trend_indicator = "📈" if trend_data['trend'] == 'improving' else "📉"
                    print(f"  {benchmark_name:25} {trend_indicator} "
                         f"Avg: {trend_data['avg_duration']:.3f}s "
                         f"Recent: {trend_data['recent_duration']:.3f}s")
        
        print("\n" + "="*60)