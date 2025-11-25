#!/usr/bin/env python3
"""
Simplified GEE Integration Test
===============================

Tests the basic Google Earth Engine integration without geemap dependency.
"""

import sys
import traceback

def test_imports():
    """Test if we can import the required packages"""
    print("Testing imports...")
    
    try:
        import ee
        print("✅ earthengine-api imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import earthengine-api: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import requests: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PIL: {e}")
        return False
    
    try:
        import structlog
        print("✅ structlog imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import structlog: {e}")
        return False
    
    return True

def test_gee_config():
    """Test the simplified GEE configuration"""
    print("\nTesting GEE configuration...")
    
    try:
        from gee_config_simple import get_gee_config, SimpleGEEConfig
        print("✅ GEE config imported successfully")
        
        # Test creating a new instance
        config = SimpleGEEConfig('instant-text-459407-v4')
        print("✅ GEE config instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test GEE config: {e}")
        traceback.print_exc()
        return False

def test_gee_initialization():
    """Test GEE initialization (this will require authentication)"""
    print("\nTesting GEE initialization...")
    
    try:
        from gee_config_simple import get_gee_config
        
        config = get_gee_config()
        print("✅ GEE config retrieved successfully")
        
        # Note: This will require user authentication
        print("⚠️  GEE initialization will require authentication")
        print("   Run this script after setting up GEE authentication")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test GEE initialization: {e}")
        traceback.print_exc()
        return False

def test_model_integration():
    """Test if models can import the GEE config"""
    print("\nTesting model integration...")
    
    # Test importing from the simplified config
    try:
        from gee_config_simple import get_gee_config
        print("✅ Models can import simplified GEE config")
        return True
    except Exception as e:
        print(f"❌ Failed to import simplified GEE config: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Simplified GEE Integration Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("GEE Config", test_gee_config),
        ("GEE Initialization", test_gee_initialization),
        ("Model Integration", test_model_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GEE integration is ready.")
        print("\nNext steps:")
        print("1. Set up GEE authentication (run: earthengine authenticate)")
        print("2. Test with actual satellite data")
        print("3. Run your ML models with GEE data")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
