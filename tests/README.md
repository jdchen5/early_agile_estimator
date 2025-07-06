# ML Project Effort Estimator - Test Suite

Comprehensive testing framework for the ML Project Effort Estimator with performance monitoring and production validation.

## Quick Start

```bash
# Install testing dependencies
pip install pytest psutil

# Run fast tests (development)
python run_tests.py --fast

# Run production validation
python run_tests.py --production

# Run complete test suite
python run_tests.py --all
```

## Test Structure

```
tests/
├── unit/                           # Fast, isolated tests (~2-3 min)
│   ├── test_config_loader.py       # Configuration loading
│   ├── test_constants.py           # Constants validation
│   ├── test_models.py              # Model loading & prediction
│   └── test_pipeline/              # Pipeline components
├── integration/                    # Component interaction tests (~5-10 min)
│   ├── test_pipeline_integration.py
│   └── test_model_pipeline_integration.py
├── e2e/                           # End-to-end scenarios (~10-15 min)
│   ├── test_single_model_prediction.py
│   ├── test_multi_model_comparison.py
│   └── test_production_workflow.py
├── benchmarks/                    # Performance monitoring (~15-30 min)
│   ├── test_performance_benchmarks.py
│   └── results/                   # Historical benchmark data
└── fixtures/                     # Test data and utilities
    ├── sample_configs/
    ├── test_data/
    └── mock_streamlit.py
```

## Test Categories

### Unit Tests
Fast, isolated tests focusing on individual components:
- **Configuration**: YAML/JSON loading, error handling
- **Models**: Loading, prediction, feature compatibility
- **Pipeline**: Individual transformers, validation
- **Utilities**: Helper functions, data validation

### Integration Tests  
Test component interactions:
- **Pipeline Integration**: UI input → model-ready features
- **Model-Pipeline**: Feature alignment and compatibility
- **Configuration**: Config loading → component setup

### End-to-End Tests
Complete workflow validation:
- **Single Model**: UI input → prediction → display
- **Multi-Model**: Comparison across multiple models
- **Production**: Full user scenarios and error handling

### Performance Benchmarks
Monitor system performance over time:
- **Response Times**: Prediction, SHAP analysis, feature preparation
- **Memory Usage**: Peak usage, leak detection, concurrent operations
- **Throughput**: Batch processing, concurrent predictions
- **Trend Analysis**: Performance regression detection

## Running Tests

### Development Workflow
```bash
# Fast feedback during development
python run_tests.py --fast

# Test specific integration
python run_tests.py --integration

# Validate end-to-end workflows
python run_tests.py --e2e
```

### Production Deployment
```bash
# Complete production validation
python run_tests.py --production

# Full test suite before deployment
python run_tests.py --all

# Performance benchmarking
python run_tests.py --benchmarks
```

### Advanced Options
```bash
# Test coverage analysis
python run_tests.py --coverage

# Stress testing
python run_tests.py --stress

# Prerequisites check only
python run_tests.py --check
```

## Performance Targets

| Metric | Target | Warning Threshold |
|--------|--------|------------------|
| Single Prediction | < 5s | > 7s |
| Multi-Model (3 models) | < 15s | > 20s |
| SHAP Analysis | < 30s | > 45s |
| Memory Usage | < 500MB | > 750MB |
| Feature Preparation | < 2s | > 5s |

## Test Data

### Sample Configurations
- **UI Inputs**: Realistic project parameters
- **ISBSG Data**: Subset of training dataset for testing
- **Edge Cases**: Boundary values, error conditions

### Mock Components
- **Streamlit**: UI component mocking for faster tests
- **Models**: Mock models for dependency-free testing
- **External APIs**: Mock external service calls

## Benchmark Results

Performance benchmarks are automatically stored in `benchmarks/results/`:

```
benchmarks/results/
├── 2024-12-19/
│   ├── benchmark_14-30-15.json    # Detailed results
│   ├── daily_summary.json         # Daily aggregation
│   └── ...
├── trends/
│   ├── response_time_trend.json   # Historical trends
│   └── memory_trend.json
└── baselines/
    └── baseline_metrics.json      # Performance baselines
```

## Continuous Monitoring

### Performance Regression Detection
- Compare current runs against historical baselines
- Alert on >20% performance degradation
- Track memory leak patterns
- Monitor concurrent operation stability

### Trend Analysis
- Daily performance summaries
- Week-over-week trend analysis
- Seasonal performance patterns
- Capacity planning metrics

## Test Configuration

### Environment Setup
```python
# pytest configuration in conftest.py
- Isolated temporary workspaces
- Automatic cleanup after tests
- Shared fixtures for common test data
- Mock Streamlit components
```

### Failure Handling
- **Missing Models**: Tests fail with clear error messages
- **Corrupted Configs**: Tests fail with validation details
- **Invalid Data**: Tests validate error handling works
- **Performance Regression**: Tests fail if targets exceeded

## Production Validation Checklist

Before production deployment, ensure:

✅ **System Health**
- [ ] All models load successfully
- [ ] Configuration files are valid
- [ ] Required data files present
- [ ] No critical errors in logs

✅ **Functional Validation**
- [ ] Single model predictions work
- [ ] Multi-model comparisons work
- [ ] Feature preparation pipeline works
- [ ] Error handling works gracefully

✅ **Performance Validation**
- [ ] Response times meet targets
- [ ] Memory usage within limits
- [ ] No memory leaks detected
- [ ] Concurrent operations stable

✅ **Data Validation**
- [ ] Predictions are reasonable ranges
- [ ] Feature counts match expectations
- [ ] Input validation works correctly
- [ ] Output formats are consistent

## Troubleshooting

### Common Issues

**Tests Skip with "Models not available"**
```bash
# Ensure models folder exists with .pkl files
ls models/*.pkl
# Copy trained models to models/ folder
```

**Configuration Errors**
```bash
# Validate configuration files
python -c "from config_loader import ConfigLoader; print(ConfigLoader.load_yaml_config('config/ui_info.yaml'))"
```

**Performance Test Failures**
```bash
# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_count()}, RAM: {psutil.virtual_memory().total/1e9:.1f}GB')"
```

### Debug Mode
```bash
# Run with verbose output
python -m pytest tests/ -v -s --tb=long

# Run specific test with debugging
python -m pytest tests/unit/test_models.py::TestModelLoading::test_load_real_model_if_available -v -s
```

## Contributing

When adding new functionality:

1. **Add Unit Tests**: Test individual components
2. **Update Integration Tests**: Test component interactions  
3. **Add E2E Scenarios**: Test complete user workflows
4. **Update Benchmarks**: Add performance tests for new features
5. **Update Documentation**: Keep this README current

### Test Writing Guidelines

- **Fast Unit Tests**: No external dependencies, < 1s each
- **Clear Test Names**: Describe what is being tested
- **Isolated Tests**: No shared state between tests
- **Realistic Data**: Use representative test inputs
- **Error Testing**: Test both success and failure cases