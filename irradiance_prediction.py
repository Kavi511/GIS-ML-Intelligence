import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import structlog
import requests
import json
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import time

# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Predefined location database for solar irradiance prediction
PREDEFINED_LOCATIONS = {
    "1": {"name": "Colombo, Sri Lanka", "coords": (6.9271, 79.8612), "climate": "tropical_monsoon"},
    "2": {"name": "Kandy, Sri Lanka", "coords": (7.2906, 80.6337), "climate": "tropical_monsoon"},
    "3": {"name": "Galle, Sri Lanka", "coords": (6.0535, 80.2210), "climate": "tropical_monsoon"},
    "4": {"name": "Jaffna, Sri Lanka", "coords": (9.6615, 80.0255), "climate": "tropical_monsoon"},
    "5": {"name": "Mumbai, India", "coords": (19.0760, 72.8777), "climate": "tropical_monsoon"},
    "6": {"name": "Chennai, India", "coords": (13.0827, 80.2707), "climate": "tropical_monsoon"},
    "7": {"name": "Kolkata, India", "coords": (22.5726, 88.3639), "climate": "tropical_monsoon"},
    "8": {"name": "Dhaka, Bangladesh", "coords": (23.8103, 90.4125), "climate": "tropical_monsoon"},
    "9": {"name": "Bangkok, Thailand", "coords": (13.7563, 100.5018), "climate": "tropical_monsoon"},
    "10": {"name": "Singapore", "coords": (1.3521, 103.8198), "climate": "tropical_rainforest"},
    "11": {"name": "Custom Location", "coords": None, "climate": "custom"}
}

# Cloud forecasting classes defined locally to avoid import issues
@dataclass
class CloudMovement:
    """Data class for cloud movement prediction"""
    velocity_x: float
    velocity_y: float
    direction: float  # in degrees
    speed: float
    confidence: float
    timestamp: datetime

@dataclass
class CloudForecast:
    """Data class for cloud forecast results"""
    current_mask: np.ndarray
    forecasted_mask: np.ndarray
    movement: CloudMovement
    time_horizon: int
    confidence: float

logger = structlog.get_logger()

@dataclass
class IrradiancePrediction:
    """Data class for irradiance prediction"""
    timestamp: datetime
    clear_sky_irradiance: float  # W/m²
    actual_irradiance: float     # W/m²
    cloud_impact: float          # Percentage reduction
    confidence: float
    weather_conditions: str
    location: Tuple[float, float]  # (lat, lon)

@dataclass
class EnergyProduction:
    """Data class for energy production prediction"""
    timestamp: datetime
    power_output: float          # kW
    energy_production: float     # kWh
    efficiency: float            # Percentage
    capacity_factor: float       # Percentage
    confidence: float
    location: Tuple[float, float]

@dataclass
class TrainingDataPoint:
    """Data class for training data points"""
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    cloud_coverage: float
    uv_index: float
    solar_radiation: float
    actual_irradiance: float
    location: Tuple[float, float]

class OpenWeatherMapAPI:
    """Enhanced OpenWeatherMap API integration for real-time weather data and training"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        if self.api_key == "YOUR_API_KEY_HERE":
            logger.warning("Please set your OpenWeatherMap API key")
        
        self.base_url = OPENWEATHER_BASE_URL
        self.session = requests.Session()
    
    def get_current_weather(self, lat: float, lon: float, units: str = "metric") -> Dict:
        """Get current weather data for a location"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': units
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            logger.info(f"Fetched current weather for {lat}, {lon}")
            
            return self._parse_current_weather(weather_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current weather: {e}")
            return self._get_fallback_weather_data()
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return self._get_fallback_weather_data()
    
    def get_forecast_weather(self, lat: float, lon: float, units: str = "metric") -> List[Dict]:
        """Get 5-day weather forecast for a location"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': units
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            forecast_data = response.json()
            logger.info(f"Fetched forecast weather for {lat}, {lon}")
            
            return self._parse_forecast_weather(forecast_data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch forecast weather: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing forecast data: {e}")
            return []
    
    def get_historical_weather(self, lat: float, lon: float, start_date: datetime, 
                              end_date: datetime, units: str = "metric") -> List[Dict]:
        """Get historical weather data for training (requires One Call API 3.0 subscription)"""
        try:
            # Note: Historical data requires One Call API 3.0 subscription
            # This is a simplified version that generates synthetic historical data
            logger.info(f"Generating historical weather data for {lat}, {lon} from {start_date} to {end_date}")
            
            historical_data = []
            current_date = start_date
            
            while current_date <= end_date:
                # Generate realistic weather data based on location and season
                weather_point = self._generate_historical_weather_point(lat, lon, current_date)
                historical_data.append(weather_point)
                current_date += timedelta(hours=1)  # Hourly data points
            
            logger.info(f"Generated {len(historical_data)} historical weather data points")
            return historical_data
            
        except Exception as e:
            logger.warning(f"Historical data generation failed: {e}")
            return []
    
    def get_solar_data(self, lat: float, lon: float, date: str = None) -> Dict:
        """Get solar irradiance data (requires One Call API 3.0 subscription)"""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # Note: Solar irradiance data requires One Call API 3.0
            # This is a simplified version - in practice you'd use the full API
            url = f"{self.base_url}/onecall"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'exclude': 'minutely,hourly,daily,alerts',
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            solar_data = response.json()
            return self._parse_solar_data(solar_data)
            
        except Exception as e:
            logger.warning(f"Solar data not available: {e}")
            return self._get_fallback_solar_data()
    
    def _generate_historical_weather_point(self, lat: float, lon: float, timestamp: datetime) -> Dict:
        """Generate realistic historical weather data point"""
        # Base values based on location and season
        day_of_year = timestamp.timetuple().tm_yday
        hour = timestamp.hour
        
        # Seasonal temperature variation
        if 6.0 <= lat <= 10.0:  # Sri Lanka
            base_temp = 28 + 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            base_humidity = 75 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        elif 13.0 <= lat <= 23.0:  # India
            base_temp = 30 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            base_humidity = 70 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        else:  # General tropical
            base_temp = 27 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            base_humidity = 70 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Diurnal variation
        temp_variation = 3 * np.sin(2 * np.pi * (hour - 6) / 24)
        humidity_variation = -10 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Add some randomness
        temp_noise = np.random.normal(0, 1)
        humidity_noise = np.random.normal(0, 3)
        
        temperature = base_temp + temp_variation + temp_noise
        humidity = np.clip(base_humidity + humidity_variation + humidity_noise, 0, 100)
        
        # Generate other weather parameters
        pressure = 1013 + np.random.normal(0, 5)
        wind_speed = max(0, np.random.exponential(2))
        wind_direction = np.random.uniform(0, 360)
        cloud_coverage = np.random.beta(2, 3) * 100  # Skewed towards lower values
        uv_index = max(0, 8 + 4 * np.sin(2 * np.pi * (hour - 12) / 24) + np.random.normal(0, 1))
        
        # Calculate solar radiation based on time and location
        solar_radiation = self._calculate_solar_radiation(lat, lon, timestamp)
        
        return {
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'cloud_coverage': cloud_coverage,
            'uv_index': uv_index,
            'solar_radiation': solar_radiation,
            'location': (lat, lon)
        }
    
    def _calculate_solar_radiation(self, lat: float, lon: float, timestamp: datetime) -> float:
        """Calculate solar radiation based on location and time"""
        # Convert to radians
        lat_rad = np.radians(lat)
        
        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday
        
        # Calculate solar declination
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 80)))
        decl_rad = np.radians(declination)
        
        # Calculate hour angle
        hour = timestamp.hour + timestamp.minute/60
        hour_angle = 15 * (hour - 12)  # Solar noon at 12:00
        hour_angle_rad = np.radians(hour_angle)
        
        # Calculate solar zenith angle
        cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) + 
                     np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad))
        zenith_angle = np.arccos(np.clip(cos_zenith, -1, 1))
        
        # Calculate air mass
        air_mass = 1 / np.cos(zenith_angle) if zenith_angle < np.pi/2 else 0
        
        # Calculate solar radiation
        if air_mass > 0:
            solar_constant = 1361  # W/m²
            atmospheric_transmittance = 0.75
            solar_radiation = solar_constant * atmospheric_transmittance ** air_mass * np.cos(zenith_angle)
            return max(0, solar_radiation)
        else:
            return 0
    
    def _parse_current_weather(self, weather_data: Dict) -> Dict:
        """Parse current weather response from OpenWeatherMap"""
        try:
            main = weather_data.get('main', {})
            weather = weather_data.get('weather', [{}])[0]
            wind = weather_data.get('wind', {})
            clouds = weather_data.get('clouds', {})
            
            parsed_data = {
                'temperature': main.get('temp', 20),
                'feels_like': main.get('feels_like', 20),
                'humidity': main.get('humidity', 50),
                'pressure': main.get('pressure', 1013),
                'wind_speed': wind.get('speed', 0),
                'wind_direction': wind.get('deg', 0),
                'cloud_coverage': clouds.get('all', 0),
                'weather_condition': weather.get('main', 'Clear'),
                'weather_description': weather.get('description', 'clear sky'),
                'visibility': weather_data.get('visibility', 10000) / 1000,  # Convert to km
                'timestamp': datetime.fromtimestamp(weather_data.get('dt', time.time())),
                'sunrise': datetime.fromtimestamp(weather_data.get('sys', {}).get('sunrise', time.time())),
                'sunset': datetime.fromtimestamp(weather_data.get('sys', {}).get('sunset', time.time())),
                'uv_index': weather_data.get('uvi', 0) if 'uvi' in weather_data else None
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing current weather: {e}")
            return self._get_fallback_weather_data()
    
    def _parse_forecast_weather(self, forecast_data: Dict) -> List[Dict]:
        """Parse forecast weather response from OpenWeatherMap"""
        try:
            forecast_list = []
            for item in forecast_data.get('list', []):
                main = item.get('main', {})
                weather = item.get('weather', [{}])[0]
                wind = item.get('wind', {})
                clouds = item.get('clouds', {})
                
                forecast_item = {
                    'timestamp': datetime.fromtimestamp(item.get('dt', time.time())),
                    'temperature': main.get('temp', 20),
                    'feels_like': main.get('feels_like', 20),
                    'humidity': main.get('humidity', 50),
                    'pressure': main.get('pressure', 1013),
                    'wind_speed': wind.get('speed', 0),
                    'wind_direction': wind.get('deg', 0),
                    'cloud_coverage': clouds.get('all', 0),
                    'weather_condition': weather.get('main', 'Clear'),
                    'weather_description': weather.get('description', 'clear sky'),
                    'pop': item.get('pop', 0)  # Probability of precipitation
                }
                
                forecast_list.append(forecast_item)
            
            return forecast_list
            
        except Exception as e:
            logger.error(f"Error parsing forecast weather: {e}")
            return []
    
    def _parse_solar_data(self, solar_data: Dict) -> Dict:
        """Parse solar irradiance data from OpenWeatherMap"""
        try:
            current = solar_data.get('current', {})
            
            solar_info = {
                'uv_index': current.get('uvi', 0),
                'solar_radiation': current.get('solar_radiation', 0),  # W/m² if available
                'timestamp': datetime.fromtimestamp(current.get('dt', time.time())),
                'cloud_coverage': current.get('clouds', 0),
                'visibility': current.get('visibility', 10000) / 1000
            }
            
            return solar_info
            
        except Exception as e:
            logger.error(f"Error parsing solar data: {e}")
            return self._get_fallback_solar_data()
    
    def _get_fallback_weather_data(self) -> Dict:
        """Fallback weather data when API fails"""
        return {
            'temperature': 25,
            'humidity': 60,
            'pressure': 1013,
            'wind_speed': 2,
            'wind_direction': 180,
            'cloud_coverage': 30,
            'weather_condition': 'Clear',
            'weather_description': 'clear sky',
            'visibility': 10,
            'timestamp': datetime.now(),
            'sunrise': datetime.now().replace(hour=6, minute=0, second=0, microsecond=0),
            'sunset': datetime.now().replace(hour=18, minute=0, second=0, microsecond=0),
            'uv_index': 5
        }
    
    def _get_fallback_solar_data(self) -> Dict:
        """Fallback solar data when API fails"""
        return {
            'uv_index': 5,
            'solar_radiation': 800,  # W/m²
            'timestamp': datetime.now(),
            'cloud_coverage': 30,
            'visibility': 10
        }

class Sentinel2WeatherDataFetcher:
    """Fetch weather data from Sentinel-2 and other sources"""
    
    def __init__(self):
        # GEE is handled by centralized configuration
        pass
    
    def fetch_sentinel2_image(self, lat: float, lon: float, date: str) -> np.ndarray:
        """Fetch Sentinel-2 image for weather analysis using centralized GEE config"""
        from gee_config_simple import get_gee_config
        
        gee_config = get_gee_config()
        satellite_data = gee_config.fetch_satellite_image(lat, lon, date, collection_type='sentinel2')
        
        if satellite_data['image'] is None:
            raise ValueError("No Sentinel-2 image found for this location and date.")
        
        return satellite_data['image']
    
    def extract_weather_features(self, image: np.ndarray) -> Dict:
        """Extract weather features from satellite image"""
        # Simple feature extraction from image
        # In practice, you'd use more sophisticated methods
        
        # Calculate brightness (proxy for cloud cover)
        brightness = np.mean(image)
        
        # Calculate contrast (proxy for weather conditions)
        contrast = np.std(image)
        
        # Simple weather classification based on image characteristics
        if brightness > 150:
            weather_condition = "clear"
            temperature = 25
            humidity = 40
        elif brightness > 100:
            weather_condition = "partly_cloudy"
            temperature = 22
            humidity = 60
        else:
            weather_condition = "cloudy"
            temperature = 18
            humidity = 80
        
        return {
            'temperature': temperature,
            'humidity': humidity,
            'pressure': 1013,  # Default atmospheric pressure
            'wind_speed': 5,   # Default wind speed
            'weather_condition': weather_condition,
            'brightness': brightness,
            'contrast': contrast
        }

class SolarIrradianceModel:
    """Solar irradiance prediction model with enhanced features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.is_trained = False
        
        # Solar constants
        self.solar_constant = 1361  # W/m²
        self.atmospheric_transmittance = 0.75
        
        # Weather data fetcher
        self.weather_fetcher = Sentinel2WeatherDataFetcher()
        
    def calculate_clear_sky_irradiance(self, latitude: float, longitude: float, 
                                     timestamp: datetime) -> float:
        """Calculate clear sky irradiance using solar geometry"""
        
        # Convert to radians
        lat_rad = np.radians(latitude)
        
        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday
        
        # Calculate solar declination
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 80)))
        decl_rad = np.radians(declination)
        
        # Calculate hour angle
        hour = timestamp.hour + timestamp.minute/60
        hour_angle = 15 * (hour - 12)  # Solar noon at 12:00
        hour_angle_rad = np.radians(hour_angle)
        
        # Calculate solar zenith angle
        cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) + 
                     np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad))
        zenith_angle = np.arccos(np.clip(cos_zenith, -1, 1))
        
        # Calculate air mass
        air_mass = 1 / np.cos(zenith_angle) if zenith_angle < np.pi/2 else 0
        
        # Calculate clear sky irradiance
        if air_mass > 0:
            # Simplified atmospheric model
            clear_sky = self.solar_constant * self.atmospheric_transmittance ** air_mass * np.cos(zenith_angle)
            return max(0, clear_sky)
        else:
            return 0
    
    def extract_features(self, cloud_mask: np.ndarray, 
                        cloud_movement: CloudMovement, 
                        weather_data: Dict,
                        latitude: float,
                        longitude: float,
                        timestamp: datetime) -> np.ndarray:
        """Extract features for irradiance prediction"""
        
        features = []
        
        # Cloud coverage features
        cloud_coverage = np.mean(cloud_mask > 0.5)
        cloud_density = np.std(cloud_mask)
        cloud_thickness = np.mean(cloud_mask[cloud_mask > 0.5]) if np.any(cloud_mask > 0.5) else 0
        
        # Cloud movement features
        velocity_magnitude = np.sqrt(cloud_movement.velocity_x**2 + cloud_movement.velocity_y**2)
        movement_direction = cloud_movement.direction
        movement_confidence = cloud_movement.confidence
        
        # Weather features
        temperature = weather_data.get('temperature', 20)
        humidity = weather_data.get('humidity', 50)
        pressure = weather_data.get('pressure', 1013)
        wind_speed = weather_data.get('wind_speed', 0)
        
        # Time features
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        month = timestamp.month
        
        # Location features
        lat_normalized = (latitude + 90) / 180  # Normalize to [0, 1]
        lon_normalized = (longitude + 180) / 360  # Normalize to [0, 1]
        
        # Solar geometry features
        clear_sky_irradiance = self.calculate_clear_sky_irradiance(latitude, longitude, timestamp)
        
        features = [
            cloud_coverage,
            cloud_density,
            cloud_thickness,
            velocity_magnitude,
            movement_direction,
            movement_confidence,
            temperature,
            humidity,
            pressure,
            wind_speed,
            hour,
            day_of_year,
            month,
            lat_normalized,
            lon_normalized,
            clear_sky_irradiance
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: List[Tuple[np.ndarray, CloudMovement, Dict, float, Tuple[float, float], datetime]]):
        """Train the irradiance prediction model"""
        
        X = []
        y = []
        
        for cloud_mask, cloud_movement, weather_data, actual_irradiance, location, timestamp in training_data:
            lat, lon = location
            features = self.extract_features(cloud_mask, cloud_movement, weather_data, lat, lon, timestamp)
            X.append(features.flatten())
            y.append(actual_irradiance)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Trained irradiance model with {len(X)} samples")
    
    def predict_irradiance(self, cloud_mask: np.ndarray, 
                          cloud_movement: CloudMovement, 
                          weather_data: Dict,
                          latitude: float, 
                          longitude: float,
                          timestamp: datetime) -> IrradiancePrediction:
        """Predict solar irradiance based on cloud conditions"""
        
        # Calculate clear sky irradiance
        clear_sky = self.calculate_clear_sky_irradiance(latitude, longitude, timestamp)
        
        if not self.is_trained:
            # Fallback to simple cloud impact model
            cloud_coverage = np.mean(cloud_mask > 0.5)
            cloud_impact = cloud_coverage * 0.8  # 80% reduction for full cloud coverage
            actual_irradiance = clear_sky * (1 - cloud_impact)
            confidence = 0.6
        else:
            # Use trained model
            features = self.extract_features(cloud_mask, cloud_movement, weather_data, latitude, longitude, timestamp)
            features_scaled = self.scaler.transform(features)
            actual_irradiance = self.model.predict(features_scaled)[0]
            confidence = 0.8
        
        # Calculate cloud impact
        cloud_impact = max(0, (clear_sky - actual_irradiance) / clear_sky) if clear_sky > 0 else 0
        
        # Determine weather conditions
        if cloud_impact < 0.1:
            weather_conditions = "clear"
        elif cloud_impact < 0.3:
            weather_conditions = "partly_cloudy"
        elif cloud_impact < 0.7:
            weather_conditions = "cloudy"
        else:
            weather_conditions = "overcast"
        
        return IrradiancePrediction(
            timestamp=timestamp,
            clear_sky_irradiance=clear_sky,
            actual_irradiance=actual_irradiance,
            cloud_impact=cloud_impact,
            confidence=confidence,
            weather_conditions=weather_conditions,
            location=(latitude, longitude)
        )

class EnergyProductionEstimator:
    """Convert irradiance predictions to energy production estimates"""
    
    def __init__(self, system_capacity: float = 1000):  # kW
        self.system_capacity = system_capacity
        self.panel_efficiency = 0.18  # 18% typical efficiency
        self.inverter_efficiency = 0.95  # 95% inverter efficiency
        self.system_losses = 0.85  # 85% system efficiency
        
    def estimate_energy_production(self, irradiance_prediction: IrradiancePrediction,
                                 system_area: float = 1000) -> EnergyProduction:
        """Estimate energy production from irradiance prediction"""
        
        # Calculate power output (kW)
        # Power = Irradiance * Area * Panel_Efficiency * Inverter_Efficiency * System_Losses
        power_output = (irradiance_prediction.actual_irradiance / 1000 *  # Convert W/m² to kW/m²
                       system_area * 
                       self.panel_efficiency * 
                       self.inverter_efficiency * 
                       self.system_losses)
        
        # Calculate energy production (kWh) for 1 hour
        energy_production = power_output * 1  # Assuming 1-hour intervals
        
        # Calculate efficiency relative to clear sky
        efficiency = (irradiance_prediction.actual_irradiance / 
                    irradiance_prediction.clear_sky_irradiance * 100) if irradiance_prediction.clear_sky_irradiance > 0 else 0
        
        # Calculate capacity factor
        capacity_factor = (power_output / self.system_capacity * 100) if self.system_capacity > 0 else 0
        
        return EnergyProduction(
            timestamp=irradiance_prediction.timestamp,
            power_output=power_output,
            energy_production=energy_production,
            efficiency=efficiency,
            capacity_factor=capacity_factor,
            confidence=irradiance_prediction.confidence,
            location=irradiance_prediction.location
        )
    
    def forecast_energy_production(self, irradiance_forecast: List[IrradiancePrediction],
                                 system_area: float = 1000) -> List[EnergyProduction]:
        """Forecast energy production for multiple time steps"""
        
        energy_forecast = []
        for irradiance_pred in irradiance_forecast:
            energy_prod = self.estimate_energy_production(irradiance_pred, system_area)
            energy_forecast.append(energy_prod)
        
        return energy_forecast

class WeatherDataProcessor:
    """Process and validate weather data for irradiance prediction"""
    
    def __init__(self):
        self.required_fields = ['temperature', 'humidity', 'pressure', 'wind_speed']
        self.weather_fetcher = Sentinel2WeatherDataFetcher()
    
    def validate_weather_data(self, weather_data: Dict) -> Dict:
        """Validate and fill missing weather data"""
        
        validated_data = weather_data.copy()
        
        # Set defaults for missing fields
        defaults = {
            'temperature': 20,
            'humidity': 50,
            'pressure': 1013,
            'wind_speed': 0
        }
        
        for field, default_value in defaults.items():
            if field not in validated_data or validated_data[field] is None:
                validated_data[field] = default_value
        
        # Add time-based features
        if 'timestamp' in validated_data:
            timestamp = validated_data['timestamp']
            validated_data['hour'] = timestamp.hour
            validated_data['day_of_year'] = timestamp.timetuple().tm_yday
        
        return validated_data
    
    def fetch_weather_from_satellite(self, lat: float, lon: float, date: str) -> Dict:
        """Fetch weather data from satellite imagery"""
        try:
            image = self.weather_fetcher.fetch_sentinel2_image(lat, lon, date)
            weather_data = self.weather_fetcher.extract_weather_features(image)
            weather_data['timestamp'] = datetime.strptime(date, '%Y-%m-%d')
            return weather_data
        except Exception as e:
            logger.warning(f"Failed to fetch weather data: {e}")
            return self.validate_weather_data({})

class SolarForecastingPipeline:
    """Complete solar forecasting pipeline integrating all models"""
    
    def __init__(self, openweather_api_key: str = None):
        self.irradiance_model = SolarIrradianceModel()
        self.energy_estimator = EnergyProductionEstimator()
        self.weather_processor = WeatherDataProcessor()
        self.openweather_api = OpenWeatherMapAPI(openweather_api_key)
        
    def run_forecast_pipeline(self, lat: float, lon: float, 
                             start_date: str = None, end_date: str = None,
                             system_area: float = 1000) -> Dict:
        """Run complete solar forecasting pipeline with OpenWeatherMap data"""
        
        try:
            # Fetch real-time weather data from OpenWeatherMap
            current_weather = self.openweather_api.get_current_weather(lat, lon)
            forecast_weather = self.openweather_api.get_forecast_weather(lat, lon)
            solar_data = self.openweather_api.get_solar_data(lat, lon)
            
            # Merge weather data
            weather_data = {**current_weather, **solar_data}
            
            # Generate cloud mask based on weather conditions
            cloud_mask = self._generate_cloud_mask_from_weather(weather_data)
            
            # Generate cloud movement based on wind data
            cloud_movement = self._generate_cloud_movement_from_weather(weather_data)
            
            # Predict irradiance
            irradiance_prediction = self.irradiance_model.predict_irradiance(
                cloud_mask=cloud_mask,
                cloud_movement=cloud_movement,
                weather_data=weather_data,
                latitude=lat,
                longitude=lon,
                timestamp=weather_data['timestamp']
            )
            
            # Estimate energy production
            energy_production = self.energy_estimator.estimate_energy_production(
                irradiance_prediction, system_area
            )
            
            # Generate forecast if we have forecast data
            irradiance_forecast = []
            energy_forecast = []
            
            if forecast_weather:
                for forecast_item in forecast_weather[:5]:  # Next 5 time steps
                    # Generate cloud mask for forecast
                    forecast_cloud_mask = self._generate_cloud_mask_from_weather(forecast_item)
                    forecast_cloud_movement = self._generate_cloud_movement_from_weather(forecast_item)
                    
                    # Predict irradiance for forecast
                    forecast_irradiance = self.irradiance_model.predict_irradiance(
                        cloud_mask=forecast_cloud_mask,
                        cloud_movement=forecast_cloud_movement,
                        weather_data=forecast_item,
                        latitude=lat,
                        longitude=lon,
                        timestamp=forecast_item['timestamp']
                    )
                    
                    irradiance_forecast.append(forecast_irradiance)
                    
                    # Estimate energy production for forecast
                    forecast_energy = self.energy_estimator.estimate_energy_production(
                        forecast_irradiance, system_area
                    )
                    energy_forecast.append(forecast_energy)
            
            return {
                'irradiance_prediction': irradiance_prediction,
                'energy_production': energy_production,
                'weather_data': weather_data,
                'cloud_movement': cloud_movement,
                'irradiance_forecast': irradiance_forecast,
                'energy_forecast': energy_forecast,
                'forecast_weather': forecast_weather[:5] if forecast_weather else []
            }
            
        except Exception as e:
            logger.error(f"Forecast pipeline failed: {e}")
            return {}
    
    def _generate_cloud_mask_from_weather(self, weather_data: Dict) -> np.ndarray:
        """Generate synthetic cloud mask based on weather conditions"""
        # Create a 256x256 cloud mask
        size = 256
        cloud_mask = np.zeros((size, size), dtype=np.float32)
        
        # Get cloud coverage percentage
        cloud_coverage = weather_data.get('cloud_coverage', 30) / 100.0
        
        if cloud_coverage > 0:
            # Generate random cloud patterns
            np.random.seed(int(weather_data.get('timestamp', datetime.now()).timestamp()))
            
            # Number of cloud clusters based on coverage
            num_clusters = int(cloud_coverage * 20)
            
            for _ in range(num_clusters):
                # Random cloud center
                center_x = np.random.randint(0, size)
                center_y = np.random.randint(0, size)
                
                # Cloud size based on humidity
                humidity = weather_data.get('humidity', 50)
                cloud_size = int(10 + humidity * 0.3)
                
                # Generate cloud shape
                for dx in range(-cloud_size, cloud_size):
                    for dy in range(-cloud_size, cloud_size):
                        x = (center_x + dx) % size
                        y = (center_y + dy) % size
                        
                        # Distance from center
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < cloud_size:
                            # Gaussian falloff
                            intensity = np.exp(-(dist**2) / (2 * (cloud_size/3)**2))
                            cloud_mask[x, y] = max(cloud_mask[x, y], intensity)
            
            # Normalize to [0, 1]
            if np.max(cloud_mask) > 0:
                cloud_mask = cloud_mask / np.max(cloud_mask)
        
        return cloud_mask
    
    def _generate_cloud_movement_from_weather(self, weather_data: Dict) -> CloudMovement:
        """Generate cloud movement based on wind data"""
        wind_speed = weather_data.get('wind_speed', 0)
        wind_direction = weather_data.get('wind_direction', 0)
        
        # Convert wind direction to radians
        wind_rad = np.radians(wind_direction)
        
        # Calculate velocity components
        velocity_x = wind_speed * np.cos(wind_rad)
        velocity_y = wind_speed * np.sin(wind_rad)
        
        # Calculate confidence based on wind speed
        confidence = min(0.9, 0.3 + wind_speed * 0.1)
        
        return CloudMovement(
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            direction=wind_direction,
            speed=wind_speed,
            confidence=confidence,
            timestamp=weather_data.get('timestamp', datetime.now())
        )

# --- Location Selection Functions ---
def get_location_selection():
    """Get location selection from user"""
    print("\n" + "="*60)
    print("📍 LOCATION SELECTION FOR SOLAR IRRADIANCE PREDICTION")
    print("="*60)
    
    print("\nAvailable Locations:")
    for key, location in PREDEFINED_LOCATIONS.items():
        if key != "11":  # Skip custom option in main list
            print(f"   {key}. {location['name']}")
    print("   11. Custom Location (Enter coordinates)")
    
    while True:
        try:
            choice = input("\nSelect location (1-11): ").strip()
            if choice in PREDEFINED_LOCATIONS:
                selected = PREDEFINED_LOCATIONS[choice]
                
                if choice == "11":  # Custom location
                    print("\nEnter custom coordinates:")
                    lat = float(input("   Latitude (e.g., 6.9271): "))
                    lon = float(input("   Longitude (e.g., 79.8612): "))
                    name = input("   Location name (optional): ")
                    if not name:
                        name = f"Custom Location ({lat:.4f}°N, {lon:.4f}°E)"
                    
                    return {
                        "name": name,
                        "coords": (lat, lon),
                        "climate": "custom"
                    }
                else:
                    return selected
            else:
                print("❌ Invalid choice. Please select 1-11.")
        except ValueError:
            print("❌ Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            exit()

def get_time_parameters():
    """Get time and date parameters from user"""
    print(f"\n⏰ TIME AND DATE PARAMETERS")
    print("="*60)
    
    try:
        # Date range selection
        start_date = input("   Start Date (YYYY-MM-DD) [default: 2024-01-01]: ").strip()
        if not start_date:
            start_date = "2024-01-01"
        
        end_date = input("   End Date (YYYY-MM-DD) [default: 2024-12-31]: ").strip()
        if not end_date:
            end_date = "2024-12-31"
        
        # Training data duration
        print(f"\n   Training Data Duration:")
        print("   1. Last 30 days")
        print("   2. Last 90 days")
        print("   3. Last 6 months")
        print("   4. Last 1 year")
        print("   5. Custom duration")
        
        duration_choice = input("   Select training duration (1-5) [default: 3]: ").strip()
        if not duration_choice:
            duration_choice = "3"
        
        # Forecast horizon
        forecast_horizon = int(input("   Forecast Horizon (hours ahead) [default: 24]: ") or "24")
        if forecast_horizon < 1 or forecast_horizon > 168:  # Max 1 week
            print("⚠️  Forecast horizon should be 1-168 hours. Using default 24")
            forecast_horizon = 24
        
        return start_date, end_date, duration_choice, forecast_horizon
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        return "2024-01-01", "2024-12-31", "3", 24

def get_system_parameters():
    """Get solar system parameters from user"""
    print(f"\n☀️  SOLAR SYSTEM PARAMETERS")
    print("="*60)
    
    try:
        system_capacity = float(input("   System Capacity (kW) [default: 1000]: ") or "1000")
        system_area = float(input("   System Area (m²) [default: 1000]: ") or "1000")
        
        # Training parameters
        print(f"\n   Model Training Parameters:")
        epochs = int(input("   Training Epochs [default: 100]: ") or "100")
        learning_rate = float(input("   Learning Rate [default: 0.001]: ") or "0.001")
        
        return system_capacity, system_area, epochs, learning_rate
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        return 1000.0, 1000.0, 100, 0.001

def display_location_insights(location_info: dict):
    """Display location-specific insights"""
    print(f"\n🌍 Location-Specific Insights:")
    lat, lon = location_info['coords']
    
    if location_info["climate"] == "tropical_monsoon":
        if 6.0 <= lat <= 10.0 and 79.0 <= lon <= 82.0:
            print("   🏝️  Sri Lanka Region - Tropical monsoon climate")
            print("   🌧️  Southwest monsoon (May-September) brings heavy rainfall")
            print("   ☀️  Northeast monsoon (December-March) brings dry weather")
            print("   🌡️  Year-round warm temperatures (25-32°C)")
        elif 13.0 <= lat <= 23.0 and 72.0 <= lon <= 90.0:
            print("   🇮🇳 Indian Subcontinent - Tropical monsoon climate")
            print("   🌧️  Southwest monsoon (June-September) brings heavy rainfall")
            print("   ☀️  Northeast monsoon (October-December) brings dry weather")
            print("   🌡️  Hot summers (30-40°C), mild winters (15-25°C)")
        elif 13.0 <= lat <= 15.0 and 100.0 <= lon <= 101.0:
            print("   🇹🇭 Thailand - Tropical monsoon climate")
            print("   🌧️  Southwest monsoon (May-October) brings heavy rainfall")
            print("   ☀️  Northeast monsoon (November-April) brings dry weather")
            print("   🌡️  Hot and humid year-round (25-35°C)")
    elif location_info["climate"] == "tropical_rainforest":
        print("   🇸🇬 Singapore - Tropical rainforest climate")
        print("   🌧️  Year-round rainfall with no distinct dry season")
        print("   🌊  Influenced by Intertropical Convergence Zone")
        print("   🌡️  Consistently warm and humid (25-32°C)")
    elif location_info["climate"] == "custom":
        if 0.0 <= lat <= 30.0 and 70.0 <= lon <= 110.0:
            print("   🌍 South Asian Region - Varied climate zones")
            print("   🌊  Influenced by Indian Ocean and Himalayas")
            print("   🌪️  Cyclone season (April-December)")
        elif 0.0 <= lat <= 30.0 and 100.0 <= lon <= 120.0:
            print("   🌏 Southeast Asian Region - Tropical climate")
            print("   🌧️  Monsoon-influenced rainfall patterns")
            print("   🌊  Influenced by Pacific Ocean and South China Sea")
        else:
            print("   🌍 General tropical/subtropical climate patterns")
            print("   🌡️  Temperature and humidity affect cloud formation")
            print("   💨 Wind patterns influence cloud movement")

def visualize_forecast_results(results: Dict):
    """Visualize solar forecasting results"""
    if not results:
        print("No results to visualize")
        return
    
    # Create subplots for different visualizations
    num_forecasts = len(results.get('irradiance_forecast', []))
    total_plots = 2 + (1 if num_forecasts > 0 else 0)
    
    fig, axes = plt.subplots(2, total_plots, figsize=(15, 10))
    
    # Current irradiance comparison
    irradiance_pred = results.get('irradiance_prediction')
    if irradiance_pred:
        axes[0, 0].bar(['Clear Sky', 'Actual'], 
                       [irradiance_pred.clear_sky_irradiance, irradiance_pred.actual_irradiance])
        axes[0, 0].set_title('Current Solar Irradiance')
        axes[0, 0].set_ylabel('Irradiance (W/m²)')
    
    # Current energy production
    energy_prod = results.get('energy_production')
    if energy_prod:
        axes[0, 1].bar(['Power Output', 'Energy Production'], 
                       [energy_prod.power_output, energy_prod.energy_production])
        axes[0, 1].set_title('Current Energy Production')
        axes[0, 1].set_ylabel('Energy (kW/kWh)')
    
    # Weather conditions
    weather_data = results.get('weather_data', {})
    if weather_data:
        weather_metrics = ['temperature', 'humidity', 'pressure']
        weather_values = [weather_data.get(metric, 0) for metric in weather_metrics]
        axes[1, 0].bar(weather_metrics, weather_values)
        axes[1, 0].set_title('Current Weather Conditions')
    
    # Cloud movement
    cloud_movement = results.get('cloud_movement')
    if cloud_movement:
        axes[1, 1].quiver(0, 0, cloud_movement.velocity_x, cloud_movement.velocity_y, 
                          scale=1, color='red', width=0.01)
        axes[1, 1].set_title(f'Cloud Movement\nDirection: {cloud_movement.direction:.1f}°')
        axes[1, 1].set_xlim(-2, 2)
        axes[1, 1].set_ylim(-2, 2)
        axes[1, 1].grid(True)
    
    # Forecast visualization if available
    if num_forecasts > 0:
        irradiance_forecast = results.get('irradiance_forecast', [])
        forecast_times = [f"T+{i+1}" for i in range(len(irradiance_forecast))]
        forecast_values = [pred.actual_irradiance for pred in irradiance_forecast]
        
        axes[0, 2].plot(forecast_times, forecast_values, 'o-', color='orange')
        axes[0, 2].set_title('Irradiance Forecast')
        axes[0, 2].set_ylabel('Irradiance (W/m²)')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    print("☀️  SOLAR IRRADIANCE PREDICTION MODEL WITH OPENWEATHERMAP API")
    print("="*60)
    
    try:
        # Get location selection
        location_info = get_location_selection()
        print(f"\n✅ Selected: {location_info['name']}")
        print(f"   Coordinates: {location_info['coords'][0]:.4f}°N, {location_info['coords'][1]:.4f}°E")
        
        # Get time parameters
        start_date, end_date, duration_choice, forecast_horizon = get_time_parameters()
        print(f"\n✅ Time Parameters:")
        print(f"   Start Date: {start_date}")
        print(f"   End Date: {end_date}")
        print(f"   Training Duration: {['30 days', '90 days', '6 months', '1 year', 'Custom'][int(duration_choice)-1]}")
        print(f"   Forecast Horizon: {forecast_horizon} hours")
        
        # Get system parameters
        system_capacity, system_area, epochs, learning_rate = get_system_parameters()
        print(f"\n✅ System Parameters:")
        print(f"   Capacity: {system_capacity} kW")
        print(f"   Area: {system_area} m²")
        print(f"   Training Epochs: {epochs}")
        print(f"   Learning Rate: {learning_rate}")
        
        # Display location insights
        display_location_insights(location_info)
        
        # Initialize pipeline with OpenWeatherMap API
        api_key = input("\n🔑 Enter your OpenWeatherMap API key (or press Enter to use demo mode): ").strip()
        if not api_key:
            api_key = None
            print("⚠️  Running in demo mode with fallback weather data")
        
        pipeline = SolarForecastingPipeline(api_key)
        
        # Extract coordinates
        lat, lon = location_info['coords']
        
        print(f"\n🔄 Running enhanced solar forecasting pipeline for {location_info['name']}...")
        print(f"📍 Location: {lat:.4f}°N, {lon:.4f}°E")
        print(f"📅 Analysis Period: {start_date} to {end_date}")
        print(f"⏰ Forecast Horizon: {forecast_horizon} hours")
        print(f"☀️  System Capacity: {system_capacity} kW")
        
        # Convert date strings to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Run forecast with enhanced parameters
        results = pipeline.run_forecast_pipeline(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            system_area=system_area
        )
        
        # Print comprehensive results
        if results:
            print("\n" + "="*60)
            print("☀️  ENHANCED SOLAR FORECASTING RESULTS")
            print("="*60)
            
            irradiance_pred = results['irradiance_prediction']
            energy_prod = results['energy_production']
            weather_data = results['weather_data']
            
            print(f"\n📍 Location: {location_info['name']}")
            print(f"🕐 Forecast Time: {irradiance_pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🌡️  Temperature: {weather_data.get('temperature', 'N/A')}°C")
            print(f"💧 Humidity: {weather_data.get('humidity', 'N/A')}%")
            print(f"☁️  Cloud Coverage: {weather_data.get('cloud_coverage', 'N/A')}%")
            print(f"💨 Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s")
            print(f"🌅 UV Index: {weather_data.get('uv_index', 'N/A')}")
            
            print(f"\n☀️  Solar Irradiance Analysis:")
            print(f"   • Clear Sky: {irradiance_pred.clear_sky_irradiance:.1f} W/m²")
            print(f"   • Actual: {irradiance_pred.actual_irradiance:.1f} W/m²")
            print(f"   • Cloud Impact: {irradiance_pred.cloud_impact:.1%}")
            print(f"   • Weather: {irradiance_pred.weather_conditions}")
            print(f"   • Confidence: {irradiance_pred.confidence:.1%}")
            
            print(f"\n⚡ Energy Production Forecast:")
            print(f"   • Power Output: {energy_prod.power_output:.1f} kW")
            print(f"   • Energy Production: {energy_prod.energy_production:.1f} kWh")
            print(f"   • System Efficiency: {energy_prod.efficiency:.1f}%")
            print(f"   • Capacity Factor: {energy_prod.capacity_factor:.1f}%")
            print(f"   • Forecast Quality: {'High' if energy_prod.confidence > 0.7 else 'Moderate' if energy_prod.confidence > 0.5 else 'Low'}")
            
            # Show forecast if available
            if results.get('irradiance_forecast'):
                print(f"\n🔮 Extended Forecast (Next {len(results['irradiance_forecast'])} time steps):")
                for i, forecast in enumerate(results['irradiance_forecast']):
                    print(f"   • T+{i+1}: {forecast.actual_irradiance:.1f} W/m² ({forecast.weather_conditions})")
            
            # Show training data summary if available
            if results.get('training_data_summary'):
                training_summary = results['training_data_summary']
                print(f"\n📚 Model Training Summary:")
                print(f"   • Training Data Points: {training_summary.get('data_points', 'N/A')}")
                print(f"   • Training Period: {training_summary.get('period', 'N/A')}")
                print(f"   • Model Accuracy: {training_summary.get('accuracy', 'N/A'):.1%}")
                print(f"   • Training Status: {training_summary.get('status', 'N/A')}")
            
            # Location-specific recommendations
            print(f"\n💡 Location-Specific Recommendations:")
            lat, lon = location_info['coords']
            
            if 6.0 <= lat <= 10.0 and 79.0 <= lon <= 82.0:  # Sri Lanka
                print("   🏝️  Sri Lanka: Consider monsoon season impacts on solar generation")
                print("   🌧️  Southwest monsoon (May-Sep) may reduce efficiency by 20-30%")
                print("   ☀️  Northeast monsoon (Dec-Mar) provides optimal solar conditions")
            elif 13.0 <= lat <= 23.0 and 72.0 <= lon <= 90.0:  # India
                print("   🇮🇳 India: High solar potential with seasonal variations")
                print("   🌡️  Summer months (Mar-Jun) may require cooling considerations")
                print("   🌧️  Monsoon season (Jun-Sep) affects daily generation patterns")
            elif 1.0 <= lat <= 2.0 and 103.0 <= lon <= 104.0:  # Singapore
                print("   🇸🇬 Singapore: Consistent year-round solar potential")
                print("   🌧️  Regular rainfall may require frequent cleaning")
                print("   🌡️  High humidity may affect panel efficiency")
            else:
                print("   🌍 General: Monitor local weather patterns for optimal performance")
                print("   📊 Regular cleaning and maintenance recommended")
                print("   🔧 Consider seasonal adjustments to system parameters")
            
            print("\n" + "="*60)
            
            # Visualize results
            visualize_forecast_results(results)
            
        else:
            print("❌ Forecast pipeline failed")
            print("💡 This can happen if:")
            print("   • API key is invalid or expired")
            print("   • Network connectivity issues")
            print("   • Insufficient weather data for the selected location/date range")
            print("   • OpenWeatherMap API service is temporarily unavailable")
            
            if api_key:
                print("\n🔍 Troubleshooting:")
                print("   • Verify your OpenWeatherMap API key is correct")
                print("   • Check your internet connection")
                print("   • Try a different location or date range")
                print("   • Ensure your API subscription supports the requested features")
            else:
                print("\n🔍 Demo Mode Limitations:")
                print("   • Using fallback weather data (less accurate)")
                print("   • Limited historical data availability")
                print("   • Consider getting an API key for full functionality")
        
    except KeyboardInterrupt:
        print("\n\n👋 User interrupted the process. Exiting...")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("💡 Please check your inputs and try again")
        print("   • Verify location coordinates are valid")
        print("   • Ensure date format is YYYY-MM-DD")
        print("   • Check system parameters are reasonable values") 