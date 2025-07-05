# test_pipeline.py - Standalone pipeline testing script
import sys
import os

# Add your project path
sys.path.append('.')

from models import test_sequential_pipeline

def main():
    print("🧪 Testing Sequential Pipeline...")
    
    # Test data matching your UI structure
    test_inputs = {
        'project_prf_year_of_project': 2023,
        'external_eef_industry_sector': 'Banking',
        'tech_tf_primary_programming_language': 'Java',
        'tech_tf_tools_used': 3,
        'project_prf_relative_size': 'M',
        'project_prf_functional_size': 500,
        'project_prf_development_type': '',
        'tech_tf_language_type': '',
        'project_prf_application_type': None,
        'external_eef_organisation_type': None,
        'tech_tf_architecture': '',
        'tech_tf_development_platform': '',
        'project_prf_team_size_group': '',
        'project_prf_max_team_size': 8,
        'tech_tf_server_roles': None,
        'tech_tf_client_roles': None,
        'tech_tf_web_development': True,
        'tech_tf_dbms_used': False,
        'process_pmf_prototyping_used': True,
        'project_prf_case_tool_used': False,
        'process_pmf_docs': 5,
        'people_prf_project_user_involvement': 3
    }
    
    # Run the test
    result = test_sequential_pipeline(test_inputs)
    
    # Print results
    print("="*60)
    print("SEQUENTIAL PIPELINE TEST RESULTS")
    print("="*60)
    
    print(f"Overall Success: {'✅ YES' if result['success'] else '❌ NO'}")
    print()
    
    print("Custom Pipeline:")
    if result['custom_pipeline']['success']:
        print(f"  ✅ Success: {result['custom_pipeline']['shape']}")
    else:
        print(f"  ❌ Failed: {result['custom_pipeline']['error']}")
    print()
    
    print("PyCaret Pipeline:")
    if result['pycaret_pipeline']['success']:
        print(f"  ✅ Success: {result['pycaret_pipeline']['shape']}")
    else:
        print(f"  ❌ Failed: {result['pycaret_pipeline']['error']}")
    print()
    
    print("Final Result:")
    if result['success']:
        print(f"  ✅ Shape: {result['final_result']['shape']}")
        print(f"  🔧 Sample Features: {result['final_result']['features']}")
    else:
        print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("="*60)

if __name__ == "__main__":
    main()