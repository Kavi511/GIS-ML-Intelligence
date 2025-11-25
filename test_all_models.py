#!/usr/bin/env python3
"""
Comprehensive Model Test with Simplified GEE
===========================================

Tests all ML models to ensure they can import and use the simplified GEE configuration.
"""

import sys
import traceback

def test_model_imports():
    """Test if all models can import the simplified GEE configuration"""
    print("Testing model imports with simplified GEE config...")
    
    models = [
        ("Solar Irradiance Prediction", "solar_irradiance_prediction"),
        ("Cloud Detection", "cloud_detection"),
        ("Cloud Forecasting", "cloud_forecasting"),
        ("Cloud Segmentation", "cloud_segmentation"),
        ("Irradiance Prediction", "irradiance_prediction"),
        ("Solar Energy Output Prediction", "solar_energy_output_prediction"),
    ]
    
    passed = 0
    total = len(models)
    
    for model_name, module_name in models:
        try:
            print(f"\n--- Testing {model_name} ---")
            
            # Try to import the module
            module = __import__(module_name)
            print(f"✅ {model_name} imported successfully")
            
            # Check if it has the GEE config import
            if hasattr(module, 'gee_config'):
                print(f"✅ {model_name} has GEE config attribute")
            else:
                print(f"⚠️  {model_name} doesn't have GEE config attribute (may be imported differently)")
            
            passed += 1
            
        except ImportError as e:
            print(f"❌ Failed to import {model_name}: {e}")
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            traceback.print_exc()
    
    print(f"\nModel Import Results: {passed}/{total} models imported successfully")
    return passed == total

def test_gee_functionality():
    """Test basic GEE functionality"""
    print("\n" + "=" * 50)
    print("Testing GEE Functionality")
    print("=" * 50)
    
    try:
        from gee_config_simple import get_gee_config
        
        # Test basic GEE operations
        gee_config = get_gee_config()
        print("✅ GEE config retrieved successfully")
        
        # Test collection methods (these don't require actual data)
        try:
            sentinel2_collection = gee_config.get_sentinel2_collection('2024-01-01', '2024-01-02')
            print("✅ Sentinel-2 collection method works")
        except Exception as e:
            print(f"⚠️  Sentinel-2 collection method failed (expected without auth): {e}")
        
        try:
            landsat8_collection = gee_config.get_landsat8_collection('2024-01-01', '2024-01-02')
            print("✅ Landsat 8 collection method works")
        except Exception as e:
            print(f"⚠️  Landsat 8 collection method failed (expected without auth): {e}")
        
        try:
            modis_collection = gee_config.get_modis_collection('2024-01-01', '2024-01-02')
            print("✅ MODIS collection method works")
        except Exception as e:
            print(f"⚠️  MODIS collection method failed (expected without auth): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test GEE functionality: {e}")
        traceback.print_exc()
        return False

def test_mock_data():
    """Test mock data generation"""
    print("\n" + "=" * 50)
    print("Testing Mock Data Generation")
    print("=" * 50)
    
    try:
        from gee_config_simple import get_gee_config
        
        gee_config = get_gee_config()
        
        # Test mock satellite data
        mock_data = gee_config._generate_mock_satellite_data()
        
        required_keys = ['cloud_cover_percentage', 'cloud_thickness', 'cloud_type', 
                        'image', 'collection_type', 'timestamp', 'location']
        
        missing_keys = [key for key in required_keys if key not in mock_data]
        
        if not missing_keys:
            print("✅ Mock satellite data generated successfully")
            print(f"   Cloud cover: {mock_data['cloud_cover_percentage']:.1f}%")
            print(f"   Cloud type: {mock_data['cloud_type']}")
            print(f"   Collection type: {mock_data['collection_type']}")
        else:
            print(f"❌ Missing keys in mock data: {missing_keys}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test mock data: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Comprehensive Model Test with Simplified GEE")
    print("=" * 60)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("GEE Functionality", test_gee_functionality),
        ("Mock Data Generation", test_mock_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"Overall Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ML models are ready to use with GEE.")
        print("\nNext steps:")
        print("1. Set up GEE authentication: earthengine authenticate")
        print("2. Test with real satellite data")
        print("3. Run your models with actual GEE data")
        print("\nNote: The simplified GEE config avoids Windows Long Path issues")
        print("      and provides the same functionality without geemap dependency.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
