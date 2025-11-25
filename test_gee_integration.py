"""
Test Google Earth Engine Integration
===================================

This script tests the GEE integration across all models to ensure
they can properly authenticate and access satellite data.
"""

import sys
import traceback
from datetime import datetime, timedelta
import structlog

# Configure logging
structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
logger = structlog.get_logger()

def test_gee_config():
    """Test the centralized GEE configuration"""
    try:
        logger.info("Testing GEE configuration...")
        
        from gee_config import get_gee_config, initialize_gee
        
        # Test default initialization
        gee_config = get_gee_config()
        logger.info(f"GEE Config initialized: {gee_config.is_initialized}")
        logger.info(f"Project ID: {gee_config.project_id}")
        
        # Test custom project initialization
        custom_config = initialize_gee('instant-text-459407-v4')
        logger.info(f"Custom GEE Config initialized: {custom_config.is_initialized}")
        
        return True
        
    except Exception as e:
        logger.error(f"GEE Config test failed: {e}")
        traceback.print_exc()
        return False

def test_satellite_data_fetch():
    """Test fetching satellite data"""
    try:
        logger.info("Testing satellite data fetch...")
        
        from gee_config import get_gee_config
        
        gee_config = get_gee_config()
        
        # Test coordinates (Colombo, Sri Lanka)
        lat, lon = 6.9271, 79.8612
        date = datetime.now().strftime('%Y-%m-%d')
        
        # Test Sentinel-2 data
        logger.info("Fetching Sentinel-2 data...")
        sentinel_data = gee_config.fetch_satellite_image(
            lat, lon, date, collection_type='sentinel2'
        )
        
        logger.info(f"Sentinel-2 data: Cloud cover {sentinel_data['cloud_cover_percentage']}%, "
                   f"Type: {sentinel_data['cloud_type']}")
        
        # Test Landsat 8 data
        logger.info("Fetching Landsat 8 data...")
        landsat_data = gee_config.fetch_satellite_image(
            lat, lon, date, collection_type='landsat8'
        )
        
        logger.info(f"Landsat 8 data: Cloud cover {landsat_data['cloud_cover_percentage']}%, "
                   f"Type: {landsat_data['cloud_type']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Satellite data fetch test failed: {e}")
        traceback.print_exc()
        return False

def test_time_series_data():
    """Test time series data retrieval"""
    try:
        logger.info("Testing time series data...")
        
        from gee_config import get_gee_config
        
        gee_config = get_gee_config()
        
        # Test coordinates
        lat, lon = 6.9271, 79.8612
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get time series data
        time_series = gee_config.get_time_series_data(
            lat, lon, start_date, end_date, 'sentinel2'
        )
        
        logger.info(f"Time series data: {len(time_series['timestamps'])} data points")
        
        return True
        
    except Exception as e:
        logger.error(f"Time series data test failed: {e}")
        traceback.print_exc()
        return False

def test_elevation_data():
    """Test elevation data retrieval"""
    try:
        logger.info("Testing elevation data...")
        
        from gee_config import get_gee_config
        
        gee_config = get_gee_config()
        
        # Test coordinates
        lat, lon = 6.9271, 79.8612
        
        elevation = gee_config.get_elevation_data(lat, lon)
        logger.info(f"Elevation at ({lat}, {lon}): {elevation:.1f} meters")
        
        return True
        
    except Exception as e:
        logger.error(f"Elevation data test failed: {e}")
        traceback.print_exc()
        return False

def test_model_integration():
    """Test that models can import and use GEE config"""
    try:
        logger.info("Testing model integration...")
        
        # Test solar irradiance model
        logger.info("Testing solar irradiance model...")
        from solar_irradiance_prediction import SolarIrradianceModel
        
        model = SolarIrradianceModel(model_type='xgboost')
        logger.info("Solar irradiance model initialized successfully")
        
        # Test cloud detection model
        logger.info("Testing cloud detection model...")
        from cloud_detection import UNet
        
        unet_model = UNet(n_channels=3, n_classes=1)
        logger.info("Cloud detection model initialized successfully")
        
        # Test cloud forecasting model
        logger.info("Testing cloud forecasting model...")
        from cloud_forecasting import ConvLSTM
        
        # Note: ConvLSTM requires device specification
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        convlstm = ConvLSTM(3, 64, 3, 2, 1, device)
        logger.info("Cloud forecasting model initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("Starting GEE integration tests...")
    
    tests = [
        ("GEE Configuration", test_gee_config),
        ("Satellite Data Fetch", test_satellite_data_fetch),
        ("Time Series Data", test_time_series_data),
        ("Elevation Data", test_elevation_data),
        ("Model Integration", test_model_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! GEE integration is working correctly.")
        return 0
    else:
        logger.error("⚠️  Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
