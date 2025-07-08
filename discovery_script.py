# discovery_script.py
"""
SHAP Analysis Migration Discovery Script
Analyzes current environment to understand file structure and resources
"""

import os
import glob
import pandas as pd
import json
from typing import Dict, List, Any

def discover_feature_importance_files():
    """Discover and analyze feature importance CSV files"""
    print("=" * 60)
    print("FEATURE IMPORTANCE FILES DISCOVERY")
    print("=" * 60)
    
    # Search for CSV files in config folder
    config_patterns = [
        "config/*feature_importance*.csv",
        "config/*importance*.csv", 
        "config/synthetic_isbsg*feature*.csv"
    ]
    
    found_files = []
    for pattern in config_patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    print(f"Found {len(found_files)} potential feature importance files:")
    for file in found_files:
        print(f"  - {file}")
    
    # Analyze each file
    file_analysis = {}
    for file_path in found_files:
        try:
            df = pd.read_csv(file_path)
            file_analysis[file_path] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'sample_data': df.head(3).to_dict('records') if len(df) > 0 else [],
                'model_name_guess': extract_model_name_from_path(file_path)
            }
            print(f"\nFile: {file_path}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Guessed model: {extract_model_name_from_path(file_path)}")
            
        except Exception as e:
            print(f"  ERROR reading {file_path}: {e}")
            file_analysis[file_path] = {'error': str(e)}
    
    return file_analysis

def extract_model_name_from_path(file_path: str) -> str:
    """Extract potential model name from file path"""
    filename = os.path.basename(file_path)
    
    # Try different patterns
    patterns = [
        r'feature_importance_(.+)\.csv',
        r'importance_(.+)\.csv', 
        r'synthetic_isbsg.*_(.+)\.csv'
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    return "unknown"

def discover_available_models():
    """Discover available models and their structure"""
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS DISCOVERY")
    print("=" * 60)
    
    try:
        from models import list_available_models, check_required_models
        
        models = list_available_models()
        model_status = check_required_models()
        
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - Technical: {model['technical_name']}")
            print(f"    Display: {model['display_name']}")
        
        print(f"\nModel status: {model_status}")
        
        return {
            'models': models,
            'status': model_status
        }
        
    except Exception as e:
        print(f"ERROR discovering models: {e}")
        return {'error': str(e)}

def discover_isbsg_integration():
    """Discover ISBSG dataset integration"""
    print("\n" + "=" * 60)
    print("ISBSG DATASET DISCOVERY")
    print("=" * 60)
    
    try:
        from models import get_isbsg_dataset_info, prepare_isbsg_sample_data
        
        # Get dataset info
        isbsg_info = get_isbsg_dataset_info()
        print("ISBSG Dataset Info:")
        for key, value in isbsg_info.items():
            print(f"  {key}: {value}")
        
        # Test sample data preparation
        sample_data = prepare_isbsg_sample_data(10)
        if sample_data is not None:
            print(f"\nISBSG Sample Data Shape: {sample_data.shape}")
            print(f"Data Type: {sample_data.dtype}")
            print(f"Value Range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
        else:
            print("\nWARNING: Could not prepare ISBSG sample data")
        
        return {
            'info': isbsg_info,
            'sample_shape': sample_data.shape if sample_data is not None else None
        }
        
    except Exception as e:
        print(f"ERROR with ISBSG integration: {e}")
        return {'error': str(e)}

def discover_current_shap_system():
    """Discover current SHAP system capabilities"""
    print("\n" + "=" * 60)
    print("CURRENT SHAP SYSTEM DISCOVERY")
    print("=" * 60)
    
    try:
        from shap_analysis import (
            get_shap_explainer, get_sample_data_info, 
            clear_explainer_cache, get_cache_info
        )
        
        # Test basic functionality
        sample_info = get_sample_data_info()
        cache_info = get_cache_info()
        
        print("Sample Data Info:")
        for key, value in sample_info.items():
            print(f"  {key}: {value}")
        
        print(f"\nCache Info: {cache_info}")
        
        return {
            'sample_info': sample_info,
            'cache_info': cache_info
        }
        
    except Exception as e:
        print(f"ERROR with current SHAP system: {e}")
        return {'error': str(e)}

def test_new_system_imports():
    """Test if new SHAP system modules can be imported"""
    print("\n" + "=" * 60)
    print("NEW SHAP SYSTEM IMPORT TEST")
    print("=" * 60)
    
    modules_to_test = [
        'shap_analysis.data_preparer',
        'shap_analysis.explainer_factory', 
        'shap_analysis.value_calculator',
        'shap_analysis.analysis_coordinator',
        'shap_analysis.ui_integration'
    ]
    
    import_results = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            import_results[module_name] = {
                'status': 'SUCCESS',
                'classes': [attr for attr in dir(module) if not attr.startswith('_')]
            }
            print(f"✓ {module_name}: SUCCESS")
            
        except Exception as e:
            import_results[module_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"✗ {module_name}: FAILED - {e}")
    
    return import_results

def create_mapping_analysis(models: List[Dict], csv_files: Dict):
    """Analyze how models map to CSV files"""
    print("\n" + "=" * 60)
    print("MODEL-TO-CSV MAPPING ANALYSIS")
    print("=" * 60)
    
    mapping_results = {}
    
    for model in models:
        tech_name = model['technical_name']
        display_name = model['display_name']
        
        # Try to find matching CSV
        potential_matches = []
        for csv_path, csv_info in csv_files.items():
            if 'error' not in csv_info:
                guessed_name = csv_info.get('model_name_guess', '')
                if tech_name in csv_path or guessed_name == tech_name:
                    potential_matches.append(csv_path)
        
        mapping_results[tech_name] = {
            'display_name': display_name,
            'csv_matches': potential_matches,
            'has_match': len(potential_matches) > 0
        }
        
        status = "✓" if potential_matches else "✗"
        print(f"{status} {tech_name} -> {potential_matches}")
    
    return mapping_results

def generate_discovery_report(results: Dict):
    """Generate comprehensive discovery report"""
    print("\n" + "=" * 80)
    print("DISCOVERY REPORT SUMMARY")
    print("=" * 80)
    
    # CSV Files Summary
    csv_files = results.get('csv_files', {})
    print(f"📄 Feature Importance Files: {len(csv_files)} found")
    
    # Models Summary  
    models_info = results.get('models', {})
    models = models_info.get('models', [])
    print(f"🤖 Available Models: {len(models)}")
    
    # Mapping Summary
    mapping = results.get('mapping', {})
    matched_models = sum(1 for info in mapping.values() if info['has_match'])
    print(f"🔗 Model-CSV Mappings: {matched_models}/{len(models)} models have matching CSV files")
    
    # ISBSG Summary
    isbsg_info = results.get('isbsg', {})
    isbsg_available = 'error' not in isbsg_info
    print(f"📊 ISBSG Integration: {'✓ Available' if isbsg_available else '✗ Issues detected'}")
    
    # New System Summary
    imports = results.get('new_system_imports', {})
    successful_imports = sum(1 for info in imports.values() if info['status'] == 'SUCCESS')
    print(f"🔧 New System Modules: {successful_imports}/{len(imports)} modules imported successfully")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS FOR PHASE 2:")
    
    if matched_models < len(models):
        print(f"  - Implement fallback feature extraction for {len(models) - matched_models} models without CSV files")
    
    if not isbsg_available:
        print("  - Fix ISBSG dataset integration issues")
    
    if successful_imports < len(imports):
        print(f"  - Fix import issues for {len(imports) - successful_imports} new system modules")
    
    print("  - Create unified CSV-to-model mapping configuration")
    print("  - Implement backward compatibility layer")

def main():
    """Run complete discovery analysis"""
    print("SHAP ANALYSIS MIGRATION - DISCOVERY PHASE")
    print("=" * 80)
    
    results = {}
    
    # Run all discovery functions
    results['csv_files'] = discover_feature_importance_files()
    results['models'] = discover_available_models()
    results['isbsg'] = discover_isbsg_integration()
    results['current_shap'] = discover_current_shap_system()
    results['new_system_imports'] = test_new_system_imports()
    
    # Create mapping analysis
    if 'models' in results and 'csv_files' in results:
        models = results['models'].get('models', [])
        csv_files = results['csv_files']
        results['mapping'] = create_mapping_analysis(models, csv_files)
    
    # Generate final report
    generate_discovery_report(results)
    
    # Save results to file
    try:
        with open('shap_migration_discovery.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Full results saved to: shap_migration_discovery.json")
    except Exception as e:
        print(f"\n❌ Could not save results: {e}")
    
    return results

if __name__ == "__main__":
    main()