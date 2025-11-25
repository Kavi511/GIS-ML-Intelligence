# Import all necessary packages
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import structlog
import ee
from datetime import datetime, timedelta
import requests
from io import BytesIO
from PIL import Image
import signal
import sys
# Remove matplotlib import to avoid issues
# import matplotlib.pyplot as plt

# Import centralized GEE configuration
from gee_config_simple import get_gee_config

# Get GEE configuration
gee_config = get_gee_config()

# Set up a logger for cleaner output
logger = structlog.get_logger()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully"""
    global shutdown_requested
    shutdown_requested = True
    logger.warning("Received interrupt signal. Gracefully shutting down...")
    sys.exit(0)

# Set up signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

# Custom location configuration for cloud forecasting
CUSTOM_LOCATION_CONFIG = {
    "default_climate": "custom",
    "coordinate_format": "decimal_degrees"
}

# --- Data Classes ---
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

# --- Model Classes ---
class ConvLSTMCell(nn.Module):
    """ConvLSTM cell with corrected device handling"""
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape, device):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        return (nn.Parameter(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device),
                nn.Parameter(torch.zeros(batch_size, hidden, shape[0], shape[1])).to(device))

class ConvLSTM(nn.Module):
    """ConvLSTM model for cloud movement prediction"""
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels, device):
        super(ConvLSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Extend kernel_size to a list if it isn't one
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = input_channels if i == 0 else self.hidden_channels[i-1]
            cell_list.append(ConvLSTMCell(cur_input_channels, self.hidden_channels[i], kernel_size[i]))
        self.cell_list = nn.ModuleList(cell_list)
        
        self.output_conv = nn.Conv2d(self.hidden_channels[-1], output_channels, 1)

    def forward(self, input_tensor, hidden_state=None):
        batch_size, seq_len, _, height, width = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width)
        
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h, c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        last_step_output = layer_output[:, -1, :, :, :]
        output = self.output_conv(last_step_output)
        return output, None

    def _init_hidden(self, batch_size, height, width):
        return [cell.init_hidden(batch_size, self.hidden_channels[i], (height, width), self.device) for i, cell in enumerate(self.cell_list)]

# --- Predictor and Data Fetcher Classes ---
class OpticalFlowPredictor:
    """Optical flow-based cloud movement prediction"""
    def __init__(self):
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        frame1_gray = (frame1 * 255).astype(np.uint8)
        frame2_gray = (frame2 * 255).astype(np.uint8)
        return cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, **self.flow_params)

    def predict_movement(self, flow: np.ndarray, timestamp: datetime) -> CloudMovement:
        velocity_x, velocity_y = np.mean(flow[:, :, 0]), np.mean(flow[:, :, 1])
        direction = (np.degrees(np.arctan2(velocity_y, velocity_x)) + 360) % 360
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        flow_magnitude = np.linalg.norm(flow, axis=2)
        confidence = 1.0 - np.clip(np.std(flow_magnitude) / (np.mean(flow_magnitude) + 1e-6), 0, 1)
        return CloudMovement(float(velocity_x), float(velocity_y), float(direction), float(speed), float(confidence), timestamp)

class Sentinel2DataFetcher:
    """Fetch Sentinel-2 satellite data or generate mock data"""
    def fetch_time_series(self, lat: float, lon: float, start_date: str, end_date: str, days_per_image: int = 2) -> List[Tuple[np.ndarray, datetime]]:
        """Fetch satellite data or generate mock data if network fails"""
        try:
            # Try to fetch real data first
            point = ee.Geometry.Point([lon, lat])
            
            # Ensure date range is valid (add 1 day if start and end are the same)
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now().date()
            
            # Validate dates are not in the future (Sentinel-2 data is only available for past dates)
            if end_dt.date() > today:
                # Cap end_date to 3 days ago to ensure data availability
                end_dt = datetime.combine(today - timedelta(days=3), datetime.min.time())
                end_date = end_dt.strftime("%Y-%m-%d")
                logger.warning(f"Future end date requested. Using {end_date} instead (3 days ago to ensure data availability).")
            elif end_dt.date() == today:
                # If today, use yesterday
                end_dt = datetime.combine(today - timedelta(days=1), datetime.min.time())
                end_date = end_dt.strftime("%Y-%m-%d")
                logger.info(f"Today's date requested. Using {end_date} instead (yesterday to ensure data availability).")
            
            if start_dt.date() > end_dt.date():
                # If start is after end, adjust start date
                start_dt = end_dt - timedelta(days=30)  # Use 30 days before end date
                start_date = start_dt.strftime("%Y-%m-%d")
                logger.warning(f"Start date after end date. Using {start_date} instead.")
            
            if start_dt == end_dt:
                end_dt = end_dt + timedelta(days=1)
                end_date = end_dt.strftime("%Y-%m-%d")
            
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(point).filterDate(start_date, end_date)
            
            # Add timeout for the collection size operation
            try:
                collection_size = collection.size().getInfo()
                if collection_size == 0:
                    logger.warning("No images found in the collection. Using mock data.")
                    return self._generate_mock_time_series(lat, lon, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to get collection size: {e}. Using mock data.")
                return self._generate_mock_time_series(lat, lon, start_date, end_date)
            
            image_list = collection.toList(collection_size)
            images_with_timestamps = []

            for i in range(min(collection_size, 11)):  # Limit to 11 images max to get 10 forecast results
                # Check for shutdown request
                if shutdown_requested:
                    logger.warning("Shutdown requested. Using mock data.")
                    return self._generate_mock_time_series(lat, lon, start_date, end_date)
                    
                try:
                    # Add timeout for each GEE operation
                    image = ee.Image(image_list.get(i))
                    
                    # Get timestamp with timeout
                    try:
                        timestamp = datetime.fromtimestamp(image.get('system:time_start').getInfo() / 1000)
                    except Exception as e:
                        logger.warning(f"Failed to get timestamp for image {i}: {e}")
                        timestamp = datetime.now()
                    
                    # Get thumbnail URL with timeout
                    try:
                        url = image.select(['B4', 'B3', 'B2']).getThumbURL({
                            'region': point.buffer(5000).bounds(), 'dimensions': 256, 'format': 'png', 'min': 0, 'max': 3000
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get thumbnail URL for image {i}: {e}")
                        continue
                    
                    # Add timeout to prevent hanging
                    try:
                        response = requests.get(url, timeout=15)  # Reduced timeout
                        if response.status_code == 200:
                            img_arr = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
                            images_with_timestamps.append((img_arr, timestamp))
                        else:
                            logger.warning(f"HTTP {response.status_code} for image {i}")
                    except requests.exceptions.Timeout:
                        logger.warning(f"Timeout while fetching image {i}. Skipping...")
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to download image {i}: {e}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Skipping an image due to fetch error: {e}")
                    continue
            
            if len(images_with_timestamps) >= 11:  # Need 11 images to generate 10 forecasts
                logger.info(f"Successfully fetched {len(images_with_timestamps)} images.")
                return images_with_timestamps
            else:
                logger.warning("Not enough real images found. Using mock data.")
                return self._generate_mock_time_series(lat, lon, start_date, end_date)
                
        except Exception as e:
            logger.warning(f"Failed to fetch real satellite data: {e}")
            logger.info("Generating mock data for demonstration...")
            return self._generate_mock_time_series(lat, lon, start_date, end_date)
    
    def _generate_mock_time_series(self, lat: float, lon: float, start_date: str, end_date: str) -> List[Tuple[np.ndarray, datetime]]:
        """Generate mock satellite images for demonstration"""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate 10 mock images over the date range for 10 forecast results
        num_images = 10
        images_with_timestamps = []
        
        for i in range(num_images):
            # Create timestamp
            timestamp = start_dt + (end_dt - start_dt) * i / (num_images - 1)
            
            # Generate mock satellite image (256x256 RGB)
            # Simulate different cloud patterns
            img_size = 256
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Base terrain (green/brown)
            img[:, :, 0] = np.random.randint(50, 150, (img_size, img_size))  # Red
            img[:, :, 1] = np.random.randint(100, 200, (img_size, img_size))  # Green
            img[:, :, 2] = np.random.randint(30, 100, (img_size, img_size))  # Blue
            
            # Add cloud patterns based on time
            cloud_cover = 0.3 + 0.4 * np.sin(i * np.pi / 2)  # Varying cloud cover
            num_clouds = int(cloud_cover * 20)
            
            for _ in range(num_clouds):
                # Random cloud position and size
                x = np.random.randint(0, img_size)
                y = np.random.randint(0, img_size)
                radius = np.random.randint(20, 60)
                
                # Create circular cloud
                for dx in range(-radius, radius):
                    for dy in range(-radius, radius):
                        if dx*dx + dy*dy <= radius*radius:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < img_size and 0 <= ny < img_size:
                                # White cloud pixels
                                img[ny, nx, 0] = min(255, img[ny, nx, 0] + 100)
                                img[ny, nx, 1] = min(255, img[ny, nx, 1] + 100)
                                img[ny, nx, 2] = min(255, img[ny, nx, 2] + 100)
            
            images_with_timestamps.append((img, timestamp))
        
        logger.info(f"Generated {len(images_with_timestamps)} mock satellite images.")
        return images_with_timestamps

# --- Main Orchestrator Class ---
class CloudForecastingModel:
    """Combined cloud forecasting model with corrected kernel_size"""
    def __init__(self, cloud_detection_model=None, model_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # **BUG FIX**: kernel_size must be a list or tuple
        self.convlstm = ConvLSTM(input_channels=1, hidden_channels=[64, 128, 64], kernel_size=[5,3,3],
                                  num_layers=3, output_channels=1, device=self.device)
        if model_path:
            self.convlstm.load_state_dict(torch.load(model_path, map_location=device))
        self.convlstm.to(device)
        self.convlstm.eval()
        
        self.optical_flow = OpticalFlowPredictor()
        self.data_fetcher = Sentinel2DataFetcher()
        self.cloud_detection_model = cloud_detection_model

    def run_forecast_pipeline(self, lat: float, lon: float, start_date: str, end_date: str, time_horizon: int = 1) -> List[CloudForecast]:
        images_with_timestamps = self.data_fetcher.fetch_time_series(lat, lon, start_date, end_date)
        if len(images_with_timestamps) < 11:
            raise ValueError("Need at least 11 images to generate 10 forecasts.")
        
        images, timestamps = zip(*images_with_timestamps)
        
        cloud_masks = []
        for img in images:
            if self.cloud_detection_model:
                # Use the real cloud detection model if provided
                mask = self.cloud_detection_model.predict(img)
            else:
                # **MOCK DATA**: Create a simple mock mask if no model is available
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray_img, 150, 1, cv2.THRESH_BINARY)
            cloud_masks.append(mask.astype(np.float32))
            
        forecasts = []
        for i in range(len(cloud_masks) - 1):
            current_mask = cloud_masks[i]
            next_mask = cloud_masks[i+1]
            timestamp = timestamps[i+1]
            
            flow = self.optical_flow.calculate_optical_flow(current_mask, next_mask)
            movement = self.optical_flow.predict_movement(flow, timestamp)
            
            # Use a simple translation model for forecasting position
            translation_matrix = np.float32([[1, 0, movement.velocity_x * time_horizon], [0, 1, movement.velocity_y * time_horizon]])
            forecasted_mask = cv2.warpAffine(current_mask, translation_matrix, (current_mask.shape[1], current_mask.shape[0]))
            
            forecasts.append(CloudForecast(current_mask, forecasted_mask, movement, time_horizon, movement.confidence))
            
        return forecasts

# --- Custom Location Selection Functions ---
def get_custom_location():
    """Get custom location coordinates from user"""
    print("\n" + "="*60)
    print("📍 CUSTOM LOCATION SELECTION FOR CLOUD FORECASTING")
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
            
            # Determine climate zone based on coordinates
            climate = determine_climate_zone(lat, lon)
            
            return {
                "name": name,
                "coords": (lat, lon),
                "climate": climate
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

def determine_climate_zone(lat: float, lon: float) -> str:
    """Determine climate zone based on coordinates"""
    abs_lat = abs(lat)
    
    if abs_lat <= 23.5:
        return "tropical"
    elif abs_lat <= 35:
        return "subtropical"
    elif abs_lat <= 60:
        return "temperate"
    else:
        return "polar"

def get_forecast_parameters():
    """Get forecast parameters from user"""
    print(f"\n🔮 FORECAST PARAMETERS")
    print("="*60)
    
    try:
        # Set default start date to 2016-01-01
        default_start_date = "2016-01-01"
        # Get current date
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        
        start_date = input(f"   Start Date (YYYY-MM-DD) [default: {default_start_date}]: ").strip()
        if not start_date:
            start_date = default_start_date
        
        end_date = input(f"   End Date (YYYY-MM-DD) [default: {today_str}]: ").strip()
        if not end_date:
            end_date = today_str
        
        time_horizon = int(input("   Time Horizon (steps ahead) [default: 1]: ") or "1")
        if time_horizon < 1 or time_horizon > 10:
            print("⚠️  Time horizon should be 1-10. Using default 1")
            time_horizon = 1
        
        # Add future prediction date
        print(f"\n🔮 FUTURE PREDICTION DATE")
        print("="*60)
        future_date = input(f"   Future Prediction Date (YYYY-MM-DD) [default: {today_str}]: ").strip()
        if not future_date:
            future_date = today_str
        
        return start_date, end_date, time_horizon, future_date
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        return "2016-01-01", today_str, 1, today_str

# --- Future Date Prediction Function ---
def predict_future_date(forecasts: List[CloudForecast], future_date: str, location_info: dict) -> dict:
    """Predict cloud conditions for a specific future date using past forecasts"""
    try:
        future_dt = datetime.strptime(future_date, "%Y-%m-%d")
        
        # Calculate average cloud cover from past forecasts
        total_cloud_cover = 0
        total_confidence = 0
        movement_directions = []
        movement_speeds = []
        
        for forecast in forecasts:
            total_cloud_cover += np.mean(forecast.current_mask) * 100
            total_confidence += forecast.confidence
            movement_directions.append(forecast.movement.direction)
            movement_speeds.append(forecast.movement.speed)
        
        avg_cloud_cover = total_cloud_cover / len(forecasts)
        avg_confidence = total_confidence / len(forecasts)
        avg_direction = sum(movement_directions) / len(movement_directions)
        avg_speed = sum(movement_speeds) / len(movement_speeds)
        
        # Add seasonal variation based on date
        month = future_dt.month
        seasonal_factor = 1.0
        
        # Adjust for seasonal patterns
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.2  # More clouds in winter
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 0.8  # Fewer clouds in summer
        elif month in [3, 4, 5]:  # Spring
            seasonal_factor = 1.1  # Moderate clouds in spring
        elif month in [9, 10, 11]:  # Fall
            seasonal_factor = 1.0  # Normal clouds in fall
        
        # Apply seasonal adjustment
        predicted_cloud_cover = min(100, avg_cloud_cover * seasonal_factor)
        
        # Determine weather condition based on cloud cover
        if predicted_cloud_cover < 20:
            weather_condition = "Clear/Sunny"
            weather_emoji = "☀️"
        elif predicted_cloud_cover < 50:
            weather_condition = "Partly Cloudy"
            weather_emoji = "⛅"
        elif predicted_cloud_cover < 80:
            weather_condition = "Cloudy"
            weather_emoji = "☁️"
        else:
            weather_condition = "Overcast/Rainy"
            weather_emoji = "🌧️"
        
        return {
            "date": future_date,
            "predicted_cloud_cover": predicted_cloud_cover,
            "weather_condition": weather_condition,
            "weather_emoji": weather_emoji,
            "avg_direction": avg_direction,
            "avg_speed": avg_speed,
            "confidence": avg_confidence,
            "seasonal_factor": seasonal_factor,
            "location": location_info
        }
        
    except Exception as e:
        logger.warning(f"Error predicting future date: {e}")
        return None

# --- Text-based Visualization Function (replaces matplotlib) ---
def display_forecast_text(forecast: CloudForecast):
    """Display cloud forecast results as text instead of visualization"""
    print(f"\n📊 FORECAST RESULTS:")
    print(f"   📅 Timestamp: {forecast.movement.timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"   🧭 Movement Direction: {forecast.movement.direction:.1f}°")
    print(f"   ⚡ Movement Speed: {forecast.movement.speed:.2f} pixels/step")
    print(f"   📈 Velocity X: {forecast.movement.velocity_x:.2f}")
    print(f"   📉 Velocity Y: {forecast.movement.velocity_y:.2f}")
    print(f"   🎯 Confidence: {forecast.confidence:.2f}")
    print(f"   ⏰ Time Horizon: {forecast.time_horizon} steps ahead")
    
    # Display cloud mask statistics
    current_cloud_cover = np.mean(forecast.current_mask) * 100
    forecasted_cloud_cover = np.mean(forecast.forecasted_mask) * 100
    print(f"   ☁️  Current Cloud Cover: {current_cloud_cover:.1f}%")
    print(f"   🔮 Forecasted Cloud Cover: {forecasted_cloud_cover:.1f}%")
    print(f"   📊 Cloud Cover Change: {forecasted_cloud_cover - current_cloud_cover:+.1f}%")

def display_future_prediction(prediction: dict):
    """Display future date prediction results"""
    if not prediction:
        print("❌ Could not generate future prediction.")
        return
    
    print(f"\n" + "="*60)
    print(f"🔮 FUTURE DATE PREDICTION")
    print("="*60)
    print(f"📅 Target Date: {prediction['date']}")
    print(f"📍 Location: {prediction['location']['name']}")
    
    lat, lon = prediction['location']['coords']
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    print(f"   Coordinates: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")
    
    print(f"\n🌤️  WEATHER PREDICTION:")
    print(f"   {prediction['weather_emoji']} Condition: {prediction['weather_condition']}")
    print(f"   ☁️  Predicted Cloud Cover: {prediction['predicted_cloud_cover']:.1f}%")
    print(f"   🎯 Confidence: {prediction['confidence']:.2f}")
    
    print(f"\n💨 CLOUD MOVEMENT PREDICTION:")
    print(f"   🧭 Average Direction: {prediction['avg_direction']:.1f}°")
    print(f"   ⚡ Average Speed: {prediction['avg_speed']:.2f} pixels/step")
    
    print(f"\n🌍 SEASONAL ANALYSIS:")
    print(f"   📊 Seasonal Factor: {prediction['seasonal_factor']:.2f}")
    if prediction['seasonal_factor'] > 1.0:
        print(f"   📈 Above average cloud cover expected for this season")
    elif prediction['seasonal_factor'] < 1.0:
        print(f"   📉 Below average cloud cover expected for this season")
    else:
        print(f"   ➡️  Normal seasonal cloud cover expected")
    
    print(f"\n💡 PREDICTION INSIGHTS:")
    if prediction['predicted_cloud_cover'] < 30:
        print(f"   ☀️  Excellent conditions for solar energy generation")
        print(f"   🏖️  Great weather for outdoor activities")
    elif prediction['predicted_cloud_cover'] < 60:
        print(f"   ⚡ Moderate solar energy generation expected")
        print(f"   🌤️  Generally pleasant weather conditions")
    elif prediction['predicted_cloud_cover'] < 80:
        print(f"   ☁️  Reduced solar energy generation likely")
        print(f"   🌧️  Possible light rain or drizzle")
    else:
        print(f"   🌧️  Heavy cloud cover - minimal solar energy generation")
        print(f"   ⚠️  Potential for significant rainfall")

def display_location_insights(location_info: Dict):
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

# --- Main Execution ---
if __name__ == "__main__":
    print("☁️  CLOUD FORECASTING MODEL WITH CUSTOM LOCATION")
    print("="*60)
    
    # Get custom location coordinates
    location_info = get_custom_location()
    print(f"\n✅ Selected: {location_info['name']}")
    lat, lon = location_info['coords']
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    print(f"   Coordinates: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")
    
    # Get forecast parameters
    start_date, end_date, time_horizon, future_date = get_forecast_parameters()
    print(f"\n✅ Forecast Parameters:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date: {end_date}")
    print(f"   Time Horizon: {time_horizon} steps")
    print(f"   Future Prediction Date: {future_date}")
    
    # Display location insights
    display_location_insights(location_info)
    
    # Initialize the main forecasting model
    # We pass `cloud_detection_model=None` so it uses the mock logic
    forecaster = CloudForecastingModel(cloud_detection_model=None)
    
    # Extract coordinates
    LATITUDE = location_info['coords'][0]
    LONGITUDE = location_info['coords'][1]
    
    print(f"\n🔄 Running cloud forecasting pipeline for {location_info['name']}...")
    print(f"📍 Location: {LATITUDE:.4f}°N, {LONGITUDE:.4f}°E")
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"⏰ Time Horizon: {time_horizon} steps ahead")
    print(f"🔮 Future Prediction Date: {future_date}")
    
    try:
        # Run the entire pipeline
        forecast_results = forecaster.run_forecast_pipeline(
            lat=LATITUDE, 
            lon=LONGITUDE,
            start_date=start_date, 
            end_date=end_date,
            time_horizon=time_horizon
        )
        
        # Display the results
        if forecast_results:
            print(f"\n" + "="*60)
            print(f"☁️  CLOUD FORECASTING RESULTS FOR {location_info['name'].upper()}")
            print("="*60)
            lat_dir = "N" if LATITUDE >= 0 else "S"
            lon_dir = "E" if LONGITUDE >= 0 else "W"
            print(f"📍 Location: {location_info['name']} ({abs(LATITUDE):.4f}°{lat_dir}, {abs(LONGITUDE):.4f}°{lon_dir})")
            print(f"📅 Forecast Period: {start_date} to {end_date}")
            print(f"⏰ Time Horizon: {time_horizon} steps ahead")
            print(f"🔮 Generated {len(forecast_results)} forecasts")
            
            print(f"\n📊 Forecast Summary:")
            for i, forecast in enumerate(forecast_results):
                print(f"   • Forecast {i+1}: {forecast.movement.timestamp.strftime('%Y-%m-%d %H:%M')}")
                print(f"     - Movement: {forecast.movement.direction:.1f}° at {forecast.movement.speed:.2f} pixels/step")
                print(f"     - Confidence: {forecast.confidence:.2f}")
            
            print(f"\n🎯 Detailed Results:")
            for i, forecast in enumerate(forecast_results):
                print(f"\n--- Forecast {i+1} ---")
                display_forecast_text(forecast)
            
            # Generate future date prediction
            print(f"\n" + "="*60)
            print(f"🔮 GENERATING FUTURE DATE PREDICTION")
            print("="*60)
            future_prediction = predict_future_date(forecast_results, future_date, location_info)
            display_future_prediction(future_prediction)
                
        else:
            print("❌ Could not generate any forecasts. Not enough images found in the date range.")
            print("💡 Try adjusting the date range or selecting a different location.")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 This can happen if:")
        print("   • No suitable images are found in GEE for the selected date range")
        print("   • The location has limited satellite coverage")
        print("   • Network connectivity issues with Google Earth Engine")
        print("   • Invalid coordinates or date format")
        
        if "Need at least 11 images" in str(e):
            print("\n🔍 Troubleshooting:")
            print("   • Try a longer date range (e.g., 6 months to 1 year)")
            print("   • Check if the location has good satellite coverage")
            print("   • Verify the coordinates are correct")
        
        lat_dir = "N" if LATITUDE >= 0 else "S"
        lon_dir = "E" if LONGITUDE >= 0 else "W"
        print(f"\n📍 Current location: {location_info['name']} ({abs(LATITUDE):.4f}°{lat_dir}, {abs(LONGITUDE):.4f}°{lon_dir})")
        print(f"📅 Date range: {start_date} to {end_date}")



