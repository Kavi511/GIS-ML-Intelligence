"""
Google Earth Engine Configuration
================================

Centralized configuration for Google Earth Engine integration across all models.
This file handles authentication, initialization, and common GEE operations.
"""

import ee
import structlog
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import warnings

logger = structlog.get_logger()

class GEEConfig:
    """Centralized Google Earth Engine configuration"""
    
    def __init__(self, project_id: str = 'instant-text-459407-v4'):
        self.project_id = project_id
        self.is_initialized = False
        self._initialize_gee()
    
    def _initialize_gee(self):
        """Initialize Google Earth Engine with authentication"""
        try:
            # Try to initialize first
            ee.Initialize()
            self.is_initialized = True
            logger.info("Google Earth Engine initialized successfully")
        except Exception as e:
            try:
                # Authenticate and initialize
                logger.info("Authenticating with Google Earth Engine...")
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
                self.is_initialized = True
                logger.info(f"Google Earth Engine initialized with project: {self.project_id}")
            except Exception as auth_error:
                logger.error(f"Failed to authenticate with GEE: {auth_error}")
                self.is_initialized = False
                raise auth_error
    
    def get_sentinel2_collection(self, 
                                start_date: str, 
                                end_date: str,
                                cloud_filter: float = 50) -> ee.ImageCollection:
        """Get Sentinel-2 surface reflectance collection"""
        if not self.is_initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        return ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
    
    def get_landsat8_collection(self,
                               start_date: str,
                               end_date: str,
                               cloud_filter: float = 20) -> ee.ImageCollection:
        """Get Landsat 8 collection"""
        if not self.is_initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_filter))
    
    def get_modis_collection(self,
                            start_date: str,
                            end_date: str) -> ee.ImageCollection:
        """Get MODIS collection for cloud analysis"""
        if not self.is_initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        return ee.ImageCollection('MODIS/006/MOD35_L2') \
            .filterDate(start_date, end_date)
    
    def fetch_satellite_image(self,
                             lat: float,
                             lon: float,
                             date: str,
                             buffer_meters: int = 2000,
                             dimensions: int = 512,
                             collection_type: str = 'sentinel2') -> Dict[str, Any]:
        """Fetch satellite image for a specific location and date"""
        if not self.is_initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        try:
            point = ee.Geometry.Point([lon, lat])
            
            if collection_type == 'sentinel2':
                collection = self.get_sentinel2_collection(date, date)
                rgb_bands = ['B4', 'B3', 'B2']  # Red, Green, Blue
            elif collection_type == 'landsat8':
                collection = self.get_landsat8_collection(date, date)
                rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']  # Red, Green, Blue
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            image = collection.first()
            if image is None:
                logger.warning(f"No {collection_type} image found for {date}")
                return self._generate_mock_satellite_data()
            
            # Get cloud information
            if collection_type == 'sentinel2':
                cloud_cover = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            else:  # Landsat
                cloud_cover = image.get('CLOUD_COVER').getInfo()
            
            # Select RGB bands
            rgb_image = image.select(rgb_bands)
            
            # Get image URL
            url = rgb_image.getThumbURL({
                'region': point.buffer(buffer_meters).bounds(),
                'dimensions': dimensions,
                'format': 'png',
                'min': 0,
                'max': 3000
            })
            
            # Download and process image
            response = requests.get(url)
            img_array = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
            
            # Analyze image for cloud characteristics
            cloud_info = self._analyze_clouds(img_array)
            
            return {
                'cloud_cover_percentage': cloud_cover,
                'cloud_thickness': cloud_info['thickness'],
                'cloud_type': cloud_info['type'],
                'image': img_array,
                'collection_type': collection_type,
                'timestamp': date,
                'location': (lat, lon)
            }
            
        except Exception as e:
            logger.error(f"Error fetching satellite image: {e}")
            return self._generate_mock_satellite_data()
    
    def _analyze_clouds(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze cloud characteristics from satellite image"""
        # Convert to grayscale
        gray = np.mean(image, axis=2)
        
        # Simple cloud detection based on brightness
        cloud_threshold = np.percentile(gray, 70)
        cloud_mask = gray > cloud_threshold
        cloud_cover = np.mean(cloud_mask) * 100
        
        # Estimate thickness based on brightness variance
        if np.any(cloud_mask):
            cloud_brightness = gray[cloud_mask]
            thickness = np.std(cloud_brightness) / (np.mean(cloud_brightness) + 1e-6)
        else:
            thickness = 0
        
        # Classify cloud type based on characteristics
        if cloud_cover < 10:
            cloud_type = 'clear'
        elif thickness > 0.3:
            cloud_type = 'cumulus'
        elif cloud_cover > 70:
            cloud_type = 'stratus'
        else:
            cloud_type = 'cirrus'
        
        return {
            'cover': cloud_cover,
            'thickness': thickness,
            'type': cloud_type
        }
    
    def _generate_mock_satellite_data(self) -> Dict[str, Any]:
        """Generate mock satellite data when GEE is unavailable"""
        return {
            'cloud_cover_percentage': np.random.uniform(0, 100),
            'cloud_thickness': np.random.uniform(0, 1),
            'cloud_type': np.random.choice(['clear', 'cumulus', 'stratus', 'cirrus']),
            'image': None,
            'collection_type': 'mock',
            'timestamp': datetime.now().strftime('%Y-%m-%d'),
            'location': (0, 0)
        }
    
    def get_time_series_data(self,
                            lat: float,
                            lon: float,
                            start_date: str,
                            end_date: str,
                            collection_type: str = 'sentinel2') -> Dict[str, Any]:
        """Get time series data for a location"""
        if not self.is_initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        try:
            point = ee.Geometry.Point([lon, lat])
            
            if collection_type == 'sentinel2':
                collection = self.get_sentinel2_collection(start_date, end_date)
                cloud_band = 'CLOUDY_PIXEL_PERCENTAGE'
            elif collection_type == 'landsat8':
                collection = self.get_landsat8_collection(start_date, end_date)
                cloud_band = 'CLOUD_COVER'
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            # Get time series of cloud cover
            time_series = collection.select(cloud_band).getRegion(point, 30).getInfo()
            
            return {
                'timestamps': [row[0] for row in time_series[1:]],
                'cloud_cover': [row[4] for row in time_series[1:]],
                'collection_type': collection_type,
                'location': (lat, lon)
            }
            
        except Exception as e:
            logger.error(f"Error getting time series data: {e}")
            return {
                'timestamps': [],
                'cloud_cover': [],
                'collection_type': collection_type,
                'location': (lat, lon)
            }
    
    def get_elevation_data(self, lat: float, lon: float) -> float:
        """Get elevation data for a location"""
        if not self.is_initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        try:
            point = ee.Geometry.Point([lon, lat])
            elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
            elevation_value = elevation.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=30
            ).get('elevation').getInfo()
            
            return float(elevation_value) if elevation_value is not None else 0.0
            
        except Exception as e:
            logger.error(f"Error getting elevation data: {e}")
            return 0.0

# Global GEE configuration instance
gee_config = GEEConfig()

def get_gee_config() -> GEEConfig:
    """Get the global GEE configuration instance"""
    return gee_config

def initialize_gee(project_id: str = 'instant-text-459407-v4') -> GEEConfig:
    """Initialize GEE with custom project ID"""
    global gee_config
    gee_config = GEEConfig(project_id)
    return gee_config
