"""
Solar Irradiance Prediction Model
================================

Purpose: Estimate how much sunlight (irradiance) reaches the ground, given the cloud cover
Input: Cloud cover %, weather data (temperature, humidity, wind)
Output: Forecasted irradiance values (W/m²)
Technology: XGBoost regression model
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import structlog
import ee
# import geemap  # Commented out to avoid dependency issues
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

# Import centralized GEE configuration
from gee_config_simple import get_gee_config

# Get GEE configuration
gee_config = get_gee_config()

@dataclass
class IrradianceFeatures:
    """Features for irradiance prediction"""
    cloud_cover_percentage: float
    cloud_thickness: float
    cloud_type: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    solar_zenith_angle: float
    day_of_year: int
    hour_of_day: int
    latitude: float
    longitude: float
    elevation: float
    timestamp: datetime

@dataclass
class IrradiancePrediction:
    """Irradiance prediction result"""
    predicted_irradiance: float
    clear_sky_irradiance: float
    cloud_impact_factor: float
    confidence: float
    features: IrradianceFeatures
    model_used: str
    timestamp: datetime

class WeatherDataFetcher:
    """Fetch weather data from various sources"""

    def __init__(self):
        self.base_weather_url = "https://api.open-meteo.com/v1/forecast"

    def fetch_weather_data(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weather data from Open-Meteo API"""
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': start_date,
                'end_date': end_date,
                'hourly': ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m'],
                'timezone': 'UTC'
            }

            response = requests.get(self.base_weather_url, params=params)
            data = response.json()

            # Convert to DataFrame
            weather_df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['hourly']['time']),
                'temperature': data['hourly']['temperature_2m'],
                'humidity': data['hourly']['relative_humidity_2m'],
                'pressure': data['hourly']['surface_pressure'],
                'wind_speed': data['hourly']['wind_speed_10m']
            })

            return weather_df

        except Exception as e:
            logger.warning(f"Failed to fetch weather data: {e}")
            return self._generate_mock_weather_data(lat, lon, start_date, end_date)

    def _generate_mock_weather_data(self, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock weather data for testing"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        timestamps = pd.date_range(start, end, freq='H')

        # Generate realistic weather patterns
        np.random.seed(42)
        n_hours = len(timestamps)

        weather_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': 25 + 5 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + np.random.normal(0, 2, n_hours),
            'humidity': 60 + 20 * np.random.random(n_hours),
            'pressure': 1013 + np.random.normal(0, 5, n_hours),
            'wind_speed': 5 + 3 * np.random.random(n_hours)
        })

        return weather_df

class SatelliteDataProcessor:
    """Process satellite data for irradiance prediction"""

    def __init__(self):
        pass

    def fetch_satellite_data(self, lat: float, lon: float, date: str) -> Dict[str, Any]:
        """Fetch satellite data for cloud analysis using centralized GEE config"""
        try:
            # Use centralized GEE configuration
            satellite_data = gee_config.fetch_satellite_image(lat, lon, date)
            
            return {
                'cloud_cover_percentage': satellite_data['cloud_cover_percentage'],
                'cloud_thickness': satellite_data['cloud_thickness'],
                'cloud_type': satellite_data['cloud_type'],
                'image': satellite_data['image']
            }

        except Exception as e:
            logger.warning(f"Failed to fetch satellite data: {e}")
            return self._generate_mock_satellite_data()

    def _generate_mock_satellite_data(self) -> Dict[str, Any]:
        """Generate mock satellite data"""
        return {
            'cloud_cover_percentage': np.random.uniform(0, 100),
            'cloud_thickness': np.random.uniform(0, 1),
            'cloud_type': np.random.choice(['clear', 'cumulus', 'stratus', 'cirrus']),
            'image': None
        }

    def _analyze_clouds(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze cloud characteristics from satellite image"""
        # Convert to grayscale and analyze
        gray = np.mean(image, axis=2)

        # Simple cloud detection based on brightness
        cloud_mask = gray > np.percentile(gray, 70)
        cloud_cover = np.mean(cloud_mask) * 100

        # Estimate thickness based on brightness variance
        if np.any(cloud_mask):
            cloud_brightness = gray[cloud_mask]
            thickness = np.std(cloud_brightness) / np.mean(cloud_brightness)
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

class SolarGeometryCalculator:
    """Calculate solar geometry parameters"""

    def __init__(self):
        self.solar_constant = 1361  # W/m²

    def calculate_solar_position(self, lat: float, lon: float, timestamp: datetime) -> Dict[str, float]:
        """Calculate solar zenith and azimuth angles"""
        # Convert to radians
        lat_rad = np.radians(lat)

        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday

        # Calculate solar declination
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))

        # Calculate equation of time correction
        B = np.radians(360/365 * (day_of_year - 1))
        equation_of_time = 4 * (0.000075 + 0.001868*np.cos(B) - 0.032077*np.sin(B) -
                               0.014615*np.cos(2*B) - 0.040849*np.sin(2*B))

        # Calculate solar time
        solar_time = timestamp.hour + timestamp.minute/60 + equation_of_time/60 + lon/15

        # Calculate hour angle
        hour_angle = 15 * (solar_time - 12)
        hour_angle_rad = np.radians(hour_angle)
        declination_rad = np.radians(declination)

        # Calculate solar zenith angle
        cos_zenith = (np.sin(lat_rad) * np.sin(declination_rad) +
                     np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))
        zenith_angle = np.arccos(np.clip(cos_zenith, -1, 1))

        # Calculate solar azimuth angle
        cos_azimuth = ((np.sin(declination_rad) * np.cos(lat_rad) -
                       np.cos(declination_rad) * np.sin(lat_rad) * np.cos(hour_angle_rad)) /
                      np.sin(zenith_angle))
        azimuth_angle = np.arccos(np.clip(cos_azimuth, -1, 1))

        # Adjust azimuth based on hour angle
        if hour_angle > 0:
            azimuth_angle = 2 * np.pi - azimuth_angle

        return {
            'zenith_angle': np.degrees(zenith_angle),
            'azimuth_angle': np.degrees(azimuth_angle),
            'elevation_angle': 90 - np.degrees(zenith_angle),
            'air_mass': 1 / np.cos(zenith_angle) if zenith_angle < np.pi/2 else 10
        }

    def calculate_clear_sky_irradiance(self, solar_position: Dict[str, float],
                                     weather_data: Dict[str, float]) -> float:
        """Calculate clear sky irradiance"""
        air_mass = solar_position['air_mass']
        elevation = solar_position['elevation_angle']

        if elevation <= 0:
            return 0

        # Linke turbidity factor (estimated)
        turbidity = 3.0  # Typical value for clear sky

        # Calculate clear sky irradiance using simplified model
        # This is a simplified version of the Bird clear sky model
        cos_zenith = np.cos(np.radians(solar_position['zenith_angle']))

        # Rayleigh scattering
        rayleigh = np.exp(-0.0903 * air_mass**0.84 * (1 + air_mass - air_mass**1.01))

        # Aerosol scattering and absorption
        aerosol = np.exp(-0.95 * turbidity * air_mass)

        # Water vapor absorption (simplified)
        water_vapor = 1 - 0.077 * (weather_data.get('humidity', 50)/100)**0.3 * air_mass**0.8

        # Clear sky irradiance
        clear_sky = (self.solar_constant * cos_zenith * rayleigh * aerosol * water_vapor)

        return max(0, clear_sky)

class FeatureEngineer:
    """Engineer features for irradiance prediction"""

    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, weather_df: pd.DataFrame,
                       satellite_data: Dict[str, Any],
                       lat: float, lon: float) -> pd.DataFrame:
        """Create comprehensive feature set"""

        features_list = []

        for idx, row in weather_df.iterrows():
            timestamp = row['timestamp']

            # Solar geometry
            solar_calc = SolarGeometryCalculator()
            solar_pos = solar_calc.calculate_solar_position(lat, lon, timestamp)

            # Clear sky irradiance
            weather_dict = {
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'pressure': row['pressure'],
                'wind_speed': row['wind_speed']
            }
            clear_sky_irr = solar_calc.calculate_clear_sky_irradiance(solar_pos, weather_dict)

            # Create feature dictionary with consistent numeric features only
            feature_dict = {
                'timestamp': timestamp,
                'cloud_cover_percentage': satellite_data['cloud_cover_percentage'],
                'cloud_thickness': satellite_data['cloud_thickness'],
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'pressure': row['pressure'],
                'wind_speed': row['wind_speed'],
                'solar_zenith_angle': solar_pos['zenith_angle'],
                'solar_elevation_angle': solar_pos['elevation_angle'],
                'air_mass': solar_pos['air_mass'],
                'clear_sky_irradiance': clear_sky_irr,
                'day_of_year': timestamp.dayofyear,
                'hour_of_day': timestamp.hour,
                'month': timestamp.month,
                'latitude': lat,
                'longitude': lon,
                'season': self._get_season(timestamp.month)
            }

            # Add cloud type encoding as numeric values
            cloud_type_map = {'clear': 0, 'cumulus': 1, 'stratus': 2, 'cirrus': 3}
            feature_dict['cloud_type_encoded'] = cloud_type_map.get(satellite_data['cloud_type'], 0)

            features_list.append(feature_dict)

        features_df = pd.DataFrame(features_list)

        # Add interaction features
        features_df['temp_humidity_interaction'] = features_df['temperature'] * features_df['humidity']
        features_df['cloud_temp_interaction'] = features_df['cloud_cover_percentage'] * features_df['temperature']

        return features_df

    def _get_season(self, month: int) -> int:
        """Convert month to season (0=Winter, 1=Spring, 2=Summer, 3=Fall)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    def prepare_training_data(self, features_df: pd.DataFrame,
                            target_column: str = 'actual_irradiance') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Remove non-feature columns
        feature_cols = [col for col in features_df.columns
                       if col not in ['timestamp', target_column, 'cloud_type']]

        # Select only numeric columns for scaling
        numeric_cols = []
        for col in feature_cols:
            if features_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)

        X = features_df[numeric_cols].values
        y = features_df[target_column].values if target_column in features_df.columns else None

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

class SolarIrradianceModel:
    """Main solar irradiance prediction model"""

    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }

        self.model = self.models[model_type]
        self.is_trained = False

        # Initialize components
        self.weather_fetcher = WeatherDataFetcher()
        self.satellite_processor = SatelliteDataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.solar_calculator = SolarGeometryCalculator()

    def generate_training_data(self, lat: float, lon: float, start_date: str,
                             end_date: str, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data"""
        np.random.seed(42)

        # Create timestamps
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        timestamps = pd.date_range(start, end, periods=n_samples)

        training_data = []

        for timestamp in timestamps:
            # Generate synthetic weather data
            base_temp = 25 + 10 * np.sin(2 * np.pi * timestamp.dayofyear / 365)
            temperature = base_temp + np.random.normal(0, 3)
            humidity = np.random.uniform(30, 90)
            pressure = 1013 + np.random.normal(0, 10)
            wind_speed = np.random.uniform(0, 15)

            # Generate synthetic cloud data
            cloud_cover = np.random.uniform(0, 100)
            cloud_thickness = np.random.uniform(0, 1)
            cloud_types = ['clear', 'cumulus', 'stratus', 'cirrus']
            cloud_type = np.random.choice(cloud_types, p=[0.4, 0.3, 0.2, 0.1])

            # Calculate solar geometry
            solar_pos = self.solar_calculator.calculate_solar_position(lat, lon, timestamp)

            # Calculate clear sky irradiance
            weather_dict = {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed
            }
            clear_sky_irr = self.solar_calculator.calculate_clear_sky_irradiance(solar_pos, weather_dict)

            # Simulate actual irradiance based on cloud cover
            cloud_impact = cloud_cover / 100 * (0.7 + 0.3 * np.random.random())
            actual_irradiance = clear_sky_irr * (1 - cloud_impact) * np.cos(np.radians(solar_pos['zenith_angle']))

            # Ensure non-negative
            actual_irradiance = max(0, actual_irradiance)

            # Cloud type encoding
            cloud_type_map = {'clear': 0, 'cumulus': 1, 'stratus': 2, 'cirrus': 3}
            cloud_type_encoded = cloud_type_map.get(cloud_type, 0)
            
            training_data.append({
                'timestamp': timestamp,
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'cloud_cover_percentage': cloud_cover,
                'cloud_thickness': cloud_thickness,
                'cloud_type_encoded': cloud_type_encoded,
                'solar_zenith_angle': solar_pos['zenith_angle'],
                'solar_elevation_angle': solar_pos['elevation_angle'],
                'air_mass': solar_pos['air_mass'],
                'clear_sky_irradiance': clear_sky_irr,
                'actual_irradiance': actual_irradiance,
                'day_of_year': timestamp.dayofyear,
                'hour_of_day': timestamp.hour,
                'month': timestamp.month,
                'latitude': lat,
                'longitude': lon,
                'season': self._get_season(timestamp.month)
            })

        training_df = pd.DataFrame(training_data)
        
        # Add interaction features
        training_df['temp_humidity_interaction'] = training_df['temperature'] * training_df['humidity']
        training_df['cloud_temp_interaction'] = training_df['cloud_cover_percentage'] * training_df['temperature']
        
        return training_df

    def _get_season(self, month: int) -> int:
        """Convert month to season (0=Winter, 1=Spring, 2=Summer, 3=Fall)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall

    def train(self, training_data: pd.DataFrame, target_column: str = 'actual_irradiance'):
        """Train the irradiance prediction model"""

        # Prepare features
        X, y = self.feature_engineer.prepare_training_data(training_data, target_column)

        if y is None:
            raise ValueError(f"Target column '{target_column}' not found in training data")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }

        logger.info(f"Model trained successfully. Test RMSE: {metrics['test_rmse']:.2f}")
        return metrics

    def predict(self, lat: float, lon: float, timestamp: datetime,
               cloud_cover: float = None, weather_data: Dict = None) -> IrradiancePrediction:
        """Predict irradiance for given conditions"""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Get or generate weather data
        if weather_data is None:
            weather_df = self.weather_fetcher.fetch_weather_data(
                lat, lon, timestamp.strftime('%Y-%m-%d'), timestamp.strftime('%Y-%m-%d')
            )
            if not weather_df.empty:
                weather_row = weather_df.iloc[0]
                weather_data = {
                    'temperature': weather_row['temperature'],
                    'humidity': weather_row['humidity'],
                    'pressure': weather_row['pressure'],
                    'wind_speed': weather_row['wind_speed']
                }
            else:
                weather_data = {
                    'temperature': 25,
                    'humidity': 60,
                    'pressure': 1013,
                    'wind_speed': 5
                }

        # Get or generate satellite data
        if cloud_cover is None:
            satellite_data = self.satellite_processor.fetch_satellite_data(
                lat, lon, timestamp.strftime('%Y-%m-%d')
            )
        else:
            satellite_data = {
                'cloud_cover_percentage': cloud_cover,
                'cloud_thickness': 0.5,
                'cloud_type': 'cumulus'
            }

        # Calculate solar geometry
        solar_pos = self.solar_calculator.calculate_solar_position(lat, lon, timestamp)
        clear_sky_irr = self.solar_calculator.calculate_clear_sky_irradiance(solar_pos, weather_data)

        # Create features
        features_df = self.feature_engineer.create_features(
            pd.DataFrame([{
                'timestamp': timestamp,
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'pressure': weather_data['pressure'],
                'wind_speed': weather_data['wind_speed']
            }]),
            satellite_data, lat, lon
        )

        # Prepare for prediction
        X, _ = self.feature_engineer.prepare_training_data(features_df)
        predicted_irradiance = self.model.predict(X)[0]

        # Calculate cloud impact
        cloud_impact = 1 - (predicted_irradiance / clear_sky_irr) if clear_sky_irr > 0 else 0
        cloud_impact = max(0, min(1, cloud_impact))

        # Calculate confidence based on prediction variance
        confidence = 0.9 - 0.3 * cloud_impact  # Higher confidence for clear skies

        # Create irradiance features object
        irradiance_features = IrradianceFeatures(
            cloud_cover_percentage=satellite_data['cloud_cover_percentage'],
            cloud_thickness=satellite_data['cloud_thickness'],
            cloud_type=satellite_data['cloud_type'],
            temperature=weather_data['temperature'],
            humidity=weather_data['humidity'],
            pressure=weather_data['pressure'],
            wind_speed=weather_data['wind_speed'],
            solar_zenith_angle=solar_pos['zenith_angle'],
            day_of_year=timestamp.timetuple().tm_yday,
            hour_of_day=timestamp.hour,
            latitude=lat,
            longitude=lon,
            elevation=0,  # Not implemented
            timestamp=timestamp
        )

        return IrradiancePrediction(
            predicted_irradiance=float(predicted_irradiance),
            clear_sky_irradiance=float(clear_sky_irr),
            cloud_impact_factor=float(cloud_impact),
            confidence=float(confidence),
            features=irradiance_features,
            model_used=self.model_type,
            timestamp=timestamp
        )

def visualize_prediction_analysis(predictions: List[IrradiancePrediction]):
    """Visualize irradiance prediction results"""
    if not predictions:
        print("No predictions to visualize")
        return

    # Extract data
    timestamps = [p.timestamp for p in predictions]
    predicted = [p.predicted_irradiance for p in predictions]
    clear_sky = [p.clear_sky_irradiance for p in predictions]
    cloud_impact = [p.cloud_impact_factor * 100 for p in predictions]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Irradiance comparison
    axes[0, 0].plot(timestamps, clear_sky, label='Clear Sky', color='orange', linewidth=2)
    axes[0, 0].plot(timestamps, predicted, label='Predicted', color='blue', linewidth=2)
    axes[0, 0].set_title('Irradiance Prediction vs Clear Sky')
    axes[0, 0].set_ylabel('Irradiance (W/m²)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Cloud impact
    axes[0, 1].plot(timestamps, cloud_impact, color='red', linewidth=2)
    axes[0, 1].set_title('Cloud Impact Factor')
    axes[0, 1].set_ylabel('Cloud Impact (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Confidence distribution
    confidence_scores = [p.confidence for p in predictions]
    axes[1, 0].hist(confidence_scores, bins=10, alpha=0.7, color='green')
    axes[1, 0].set_title('Prediction Confidence Distribution')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Frequency')

    # Feature importance (if available)
    if hasattr(predictions[0], 'features'):
        cloud_covers = [p.features.cloud_cover_percentage for p in predictions]
        temperatures = [p.features.temperature for p in predictions]

        axes[1, 1].scatter(cloud_covers, predicted, alpha=0.6, color='purple')
        axes[1, 1].set_title('Cloud Cover vs Predicted Irradiance')
        axes[1, 1].set_xlabel('Cloud Cover (%)')
        axes[1, 1].set_ylabel('Predicted Irradiance (W/m²)')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== Irradiance Prediction Summary ===")
    print(f"Total predictions: {len(predictions)}")
    print(f"Average predicted irradiance: {np.mean(predicted):.1f} W/m²")
    print(f"Average clear sky irradiance: {np.mean(clear_sky):.1f} W/m²")
    print(f"Average cloud impact: {np.mean(cloud_impact):.1f}%")
    print(f"Average confidence: {np.mean(confidence_scores):.2f}")

def get_custom_location():
    """Get custom location coordinates from user"""
    print("\n" + "="*60)
    print("📍 CUSTOM LOCATION SELECTION FOR SOLAR IRRADIANCE PREDICTION")
    print("="*60)
    
    print("\n🌍 Enter precise coordinates for your location:")
    print("   📍 Latitude (North-South position):")
    print("      • Range: -90° to +90°")
    print("      • Positive (+) = North of Equator")
    print("      • Negative (-) = South of Equator")
    print("      • 0° = Equator line")
    print("   📍 Longitude (East-West position):")
    print("      • Range: -180° to +180°")
    print("      • Positive (+) = East of Prime Meridian")
    print("      • Negative (-) = West of Prime Meridian")
    print("      • 0° = Prime Meridian (Greenwich, London)")
    print("   📍 Format: Decimal degrees (e.g., 51.5074, -0.1278 for London)")
    
    while True:
        try:
            print("\n📍 COORDINATE INPUT:")
            lat_input = input("   Latitude (decimal degrees): ").strip()
            lon_input = input("   Longitude (decimal degrees): ").strip()
            
            # Convert to float and validate
            lat = float(lat_input)
            lon = float(lon_input)
            
            # Validate coordinate ranges with detailed feedback
            if not (-90 <= lat <= 90):
                print("❌ Latitude must be between -90° and +90°")
                print("   • -90° = South Pole")
                print("   • 0° = Equator")
                print("   • +90° = North Pole")
                continue
            if not (-180 <= lon <= 180):
                print("❌ Longitude must be between -180° and +180°")
                print("   • -180° = International Date Line (West)")
                print("   • 0° = Prime Meridian (Greenwich)")
                print("   • +180° = International Date Line (East)")
                continue
            
            # Get location name
            name = input("   Location name (optional): ").strip()
            if not name:
                # Generate descriptive name based on coordinates with hemisphere info
                lat_dir = "N" if lat >= 0 else "S"
                lon_dir = "E" if lon >= 0 else "W"
                
                # Add hemisphere descriptions
                lat_hemisphere = "Northern Hemisphere" if lat > 0 else "Southern Hemisphere" if lat < 0 else "Equator"
                lon_hemisphere = "Eastern Hemisphere" if lon > 0 else "Western Hemisphere" if lon < 0 else "Prime Meridian"
                
                name = f"Location ({abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}) - {lat_hemisphere}, {lon_hemisphere}"
            
            return {
                "name": name,
                "coords": (lat, lon)
            }
            
        except ValueError:
            print("❌ Invalid input. Please enter valid decimal numbers.")
            print("   📍 Examples:")
            print("      • London: 51.5074°N, -0.1278°W")
            print("      • New York: 40.7128°N, -74.0060°W")
            print("      • Tokyo: 35.6762°N, 139.6503°E")
            print("      • Sydney: -33.8688°S, 151.2093°E")
            print("      • Rio de Janeiro: -22.9068°S, -43.1729°W")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            exit()

def get_time_parameters():
    """Get time and date parameters from user"""
    print(f"\n⏰ TIME AND DATE PARAMETERS")
    print("="*60)
    
    try:
        # Set default start date to 2016-01-01
        default_start_date = "2016-01-01"
        # Get current date
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        
        # Date range selection
        start_date = input(f"   Start Date (YYYY-MM-DD) [default: {default_start_date}]: ").strip()
        if not start_date:
            start_date = default_start_date
        
        end_date = input(f"   End Date (YYYY-MM-DD) [default: {today_str}]: ").strip()
        if not end_date:
            end_date = today_str
        
        # Number of prediction hours
        hours_ahead = input("   Hours to predict ahead [default: 24]: ").strip()
        if not hours_ahead:
            hours_ahead = 24
        else:
            hours_ahead = int(hours_ahead)
            if hours_ahead < 1 or hours_ahead > 168:  # Max 1 week
                print("⚠️  Hours should be 1-168. Using default 24")
                hours_ahead = 24
        
        return start_date, end_date, hours_ahead
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        return "2016-01-01", today_str, 24

# Main execution
if __name__ == "__main__":
    print("☀️  SOLAR IRRADIANCE PREDICTION MODEL WITH CUSTOM LOCATION")
    print("="*60)
    
    # Get custom location coordinates
    location_info = get_custom_location()
    print(f"\n✅ Selected: {location_info['name']}")
    lat, lon = location_info['coords']
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    print(f"   Coordinates: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")
    
    # Get time parameters
    start_date, end_date, hours_ahead = get_time_parameters()
    print(f"\n✅ Time Parameters:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date: {end_date}")
    print(f"   Hours Ahead: {hours_ahead}")
    
    # Initialize model
    print(f"\n🔧 Initializing solar irradiance model...")
    irradiance_model = SolarIrradianceModel(model_type='xgboost')

    # Generate training data
    print("📊 Generating training data...")
    training_data = irradiance_model.generate_training_data(
        lat=lat, lon=lon, start_date=start_date, end_date=end_date, n_samples=2000
    )

    # Train model
    print("🎯 Training model...")
    metrics = irradiance_model.train(training_data)
    print(f"✅ Training completed. Test RMSE: {metrics['test_rmse']:.2f} W/m²")

    # Make predictions
    print(f"🔮 Making predictions for next {hours_ahead} hours...")
    predictions = []
    for i in range(hours_ahead):
        timestamp = datetime.now() + timedelta(hours=i)
        prediction = irradiance_model.predict(lat, lon, timestamp)
        predictions.append(prediction)

    # Display results
    print(f"\n" + "="*60)
    print(f"☀️  SOLAR IRRADIANCE PREDICTION RESULTS FOR {location_info['name'].upper()}")
    print("="*60)
    print(f"📍 Location: {location_info['name']} ({abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir})")
    print(f"📅 Analysis Period: {start_date} to {end_date}")
    print(f"⏰ Prediction Horizon: {hours_ahead} hours ahead")
    
    # Visualize results
    visualize_prediction_analysis(predictions)
