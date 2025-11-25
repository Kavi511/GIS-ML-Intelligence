import ee

# Import centralized GEE configuration
from gee_config_simple import get_gee_config

# Get GEE configuration
gee_config = get_gee_config()

# Custom location configuration for cloud detection
CUSTOM_LOCATION_CONFIG = {
    "default_climate": "custom",
    "coordinate_format": "decimal_degrees"
}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import structlog
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set up logger
logger = structlog.get_logger()

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class CloudDetectionModel:
    def __init__(self, model_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = UNet(n_channels=3, n_classes=1, bilinear=True)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        transformed = self.transform(image=image)
        return transformed['image'].unsqueeze(0).to(self.device)
    def predict(self, image: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output)
            prediction = prediction.cpu().numpy()[0,0]
        return prediction
    def predict_batch(self, images: list) -> list:
        return [self.predict(img) for img in images]

class CloudClassifier:
    def __init__(self):
        self.cloud_types = {
            'cumulus': 'fair_weather',
            'stratus': 'overcast',
            'cirrus': 'high_altitude',
            'cumulonimbus': 'storm'
        }
    def classify_clouds(self, cloud_mask: np.ndarray, image: np.ndarray) -> dict:
        cloud_coverage = np.mean(cloud_mask > 0.5)
        cloud_density = np.std(cloud_mask)
        if cloud_coverage < 0.1:
            cloud_type = 'clear'
        elif cloud_coverage < 0.3:
            cloud_type = 'scattered'
        elif cloud_coverage < 0.7:
            cloud_type = 'broken'
        else:
            cloud_type = 'overcast'
        return {
            'cloud_type': cloud_type,
            'cloud_coverage': float(cloud_coverage),
            'cloud_density': float(cloud_density),
            'confidence': 0.8
        }

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def fetch_sentinel2_image(lat, lon, start_date, end_date):
    """Fetch high-quality Sentinel-2 satellite image for cloud detection"""
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # Ensure date range is valid (add 1 day if start and end are the same)
        from datetime import datetime, timedelta
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
        
        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))  # Lower cloud threshold for better quality
        
        # Check if collection is empty before calling first()
        collection_size = 0
        try:
            collection_size = collection.size().getInfo()
            if collection_size == 0:
                # Try with higher cloud threshold if no image found
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterBounds(point) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                collection_size = collection.size().getInfo()
                if collection_size == 0:
                    raise ValueError("No Sentinel-2 image found for this location and date range.")
        except Exception as e:
            if "Empty date ranges" in str(e) or collection_size == 0:
                raise ValueError("No Sentinel-2 image found for this location and date range.")
            raise
        
        image = collection.first()
        if image is None:
            raise ValueError("No Sentinel-2 image found for this location and date range.")
        
        rgb_image = image.select(['B4', 'B3', 'B2'])
        url = rgb_image.getThumbURL({
            'region': point.buffer(3000).bounds(),  # Larger area for better context
            'dimensions': 1024,  # Higher resolution
            'format': 'png',
            'min': 0,
            'max': 3000
        })
        
        logger.info(f"Fetching image from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        logger.info(f"Successfully fetched image: {img.shape}")
        
        # Enhance image quality
        img = enhance_image_quality(img)
        
        return img
        
    except Exception as e:
        logger.error(f"Failed to fetch Sentinel-2 image: {e}")
        raise

def enhance_image_quality(img: np.ndarray) -> np.ndarray:
    """Enhance satellite image quality for better visualization"""
    try:
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Apply contrast enhancement
        p2, p98 = np.percentile(img_float, (2, 98))
        img_enhanced = np.clip((img_float - p2) / (p98 - p2), 0, 1)
        
        # Apply slight sharpening
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(img_enhanced, sigma=0.5)
        sharpened = np.clip(img_enhanced + 0.3 * (img_enhanced - blurred), 0, 1)
        
        # Convert back to uint8
        img_enhanced = (sharpened * 255).astype(np.uint8)
        
        return img_enhanced
        
    except Exception as e:
        logger.warning(f"Image enhancement failed: {e}, returning original image")
        return img

# --- Custom Location Selection Functions ---
def get_custom_location():
    """Get custom location coordinates from user"""
    print("\n" + "="*60)
    print("📍 CUSTOM LOCATION SELECTION FOR CLOUD DETECTION")
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
        
        # Time of day preference
        print("\n   Time of Day Preference:")
        print("   1. Morning (6:00-12:00)")
        print("   2. Afternoon (12:00-18:00)")
        print("   3. Evening (18:00-24:00)")
        print("   4. Any time (default)")
        
        time_choice = input("   Select time preference (1-4) [default: 4]: ").strip()
        if not time_choice:
            time_choice = "4"
        
        # Cloud coverage threshold
        cloud_threshold = input("   Cloud Coverage Threshold % [default: 50]: ").strip()
        if not cloud_threshold:
            cloud_threshold = 50
        else:
            cloud_threshold = int(cloud_threshold)
            if cloud_threshold < 0 or cloud_threshold > 100:
                print("⚠️  Cloud threshold should be 0-100%. Using default 50%")
                cloud_threshold = 50
        
        return start_date, end_date, time_choice, cloud_threshold
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        return "2016-01-01", today_str, "4", 50

def display_location_insights(location_info: dict):
    """Display location-specific insights based on coordinates"""
    print(f"\n🌍 Location-Specific Insights:")
    lat, lon = location_info['coords']
    climate = location_info['climate']
    
    # Display detailed coordinate information
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    print(f"   📍 Geographic Position:")
    print(f"      • Latitude: {abs(lat):.4f}°{lat_dir} ({lat:.4f}°)")
    print(f"      • Longitude: {abs(lon):.4f}°{lon_dir} ({lon:.4f}°)")
    
    # Hemisphere information
    lat_hemisphere = "Northern Hemisphere" if lat > 0 else "Southern Hemisphere" if lat < 0 else "Equator"
    lon_hemisphere = "Eastern Hemisphere" if lon > 0 else "Western Hemisphere" if lon < 0 else "Prime Meridian"
    print(f"      • Hemispheres: {lat_hemisphere}, {lon_hemisphere}")
    
    # Distance from reference lines
    if lat != 0:
        equator_distance = f"{abs(lat):.1f}° {'north' if lat > 0 else 'south'} of Equator"
        print(f"      • Position: {equator_distance}")
    else:
        print(f"      • Position: On the Equator")
        
    if lon != 0:
        meridian_distance = f"{abs(lon):.1f}° {'east' if lon > 0 else 'west'} of Prime Meridian"
        print(f"      • Position: {meridian_distance}")
    else:
        print(f"      • Position: On the Prime Meridian")
    
    # Climate zone insights
    if climate == "tropical":
        print("   🌴 Tropical Climate Zone (0°-23.5° latitude)")
        print("   🌡️  High temperatures year-round (20-35°C)")
        print("   🌧️  High humidity and frequent rainfall")
        print("   ☁️  Common cloud types: Cumulus, Cumulonimbus, Cirrus")
        print("   🌊 Influenced by Intertropical Convergence Zone")
    elif climate == "subtropical":
        print("   🌞 Subtropical Climate Zone (23.5°-35° latitude)")
        print("   🌡️  Warm to hot summers, mild winters")
        print("   🌧️  Seasonal rainfall patterns")
        print("   ☁️  Common cloud types: Stratus, Cumulus, Altocumulus")
        print("   💨 Influenced by subtropical high-pressure systems")
    elif climate == "temperate":
        print("   🍂 Temperate Climate Zone (35°-60° latitude)")
        print("   🌡️  Four distinct seasons with moderate temperatures")
        print("   🌧️  Variable precipitation throughout the year")
        print("   ☁️  Common cloud types: Stratus, Nimbostratus, Altostratus")
        print("   💨 Influenced by mid-latitude weather systems")
    elif climate == "polar":
        print("   ❄️  Polar Climate Zone (60°-90° latitude)")
        print("   🌡️  Cold temperatures year-round")
        print("   🌨️  Low precipitation, mostly snow")
        print("   ☁️  Common cloud types: Cirrus, Cirrostratus")
        print("   💨 Influenced by polar high-pressure systems")
    
    # Geographic region insights
    if 0 <= lat <= 30 and 70 <= lon <= 110:
        print("   🌍 South Asian Region")
        print("   🌊 Influenced by Indian Ocean monsoon patterns")
        print("   🌪️  Cyclone season typically April-December")
    elif 0 <= lat <= 30 and 100 <= lon <= 120:
        print("   🌏 Southeast Asian Region")
        print("   🌊 Influenced by Pacific Ocean and South China Sea")
        print("   🌧️  Monsoon-influenced rainfall patterns")
    elif 30 <= lat <= 60 and -80 <= lon <= -60:
        print("   🌎 North American Region")
        print("   🌊 Influenced by Atlantic and Pacific Oceans")
        print("   💨 Affected by jet stream patterns")
    elif 30 <= lat <= 60 and -10 <= lon <= 40:
        print("   🇪🇺 European Region")
        print("   🌊 Influenced by Atlantic Ocean and Mediterranean Sea")
        print("   💨 Affected by westerly wind patterns")
    else:
        print("   🌍 General geographic patterns")
        print("   🌡️  Temperature and humidity affect cloud formation")
        print("   💨 Wind patterns influence cloud movement")

def run_cloud_detection_pipeline():
    """Run the complete cloud detection pipeline with custom location input"""
    print("☁️  CLOUD DETECTION MODEL WITH CUSTOM LOCATION")
    print("="*60)
    
    try:
        # Get custom location coordinates
        location_info = get_custom_location()
        print(f"\n✅ Selected: {location_info['name']}")
        lat, lon = location_info['coords']
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        print(f"   Coordinates: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}")
        
        # Get time parameters
        start_date, end_date, time_choice, cloud_threshold = get_time_parameters()
        print(f"\n✅ Time Parameters:")
        print(f"   Start Date: {start_date}")
        print(f"   End Date: {end_date}")
        print(f"   Time Preference: {['Morning', 'Afternoon', 'Evening', 'Any time'][int(time_choice)-1]}")
        print(f"   Cloud Threshold: {cloud_threshold}%")
        
        # Display location insights
        display_location_insights(location_info)
        
        # Extract coordinates
        lat, lon = location_info['coords']
        
        print(f"\n🔄 Running cloud detection pipeline for {location_info['name']}...")
        print(f"📍 Location: {lat:.4f}°N, {lon:.4f}°E")
        print(f"📅 Date Range: {start_date} to {end_date}")
        print(f"☁️  Cloud Threshold: {cloud_threshold}%")
        
        # Initialize models
        cloud_model = CloudDetectionModel(model_path=None)  # untrained model
        classifier = CloudClassifier()
        
        # Fetch satellite image
        print(f"\n📡 Fetching Sentinel-2 satellite image...")
        img = fetch_sentinel2_image(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date
        )
        
        # Run cloud detection
        print(f"🔍 Running cloud detection model...")
        mask = cloud_model.predict(img)
        
        # Classify clouds
        print(f"🏷️  Classifying cloud types...")
        result = classifier.classify_clouds(mask, img)
        
        # Display results
        print(f"\n" + "="*60)
        print(f"☁️  CLOUD DETECTION RESULTS FOR {location_info['name'].upper()}")
        print("="*60)
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        print(f"📍 Location: {location_info['name']} ({abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir})")
        print(f"📅 Analysis Period: {start_date} to {end_date}")
        print(f"⏰ Time Preference: {['Morning', 'Afternoon', 'Evening', 'Any time'][int(time_choice)-1]}")
        
        print(f"\n📊 Cloud Detection Results:")
        print(f"   • Cloud Type: {result['cloud_type']}")
        print(f"   • Cloud Coverage: {result['cloud_coverage']:.1%}")
        print(f"   • Cloud Density: {result['cloud_density']:.3f}")
        print(f"   • Confidence: {result['confidence']:.1%}")
        
        # Additional analysis
        print(f"\n🔍 Detailed Analysis:")
        if result['cloud_coverage'] < 0.1:
            print("   ☀️  Clear skies - Excellent visibility conditions")
        elif result['cloud_coverage'] < 0.3:
            print("   🌤️  Scattered clouds - Good visibility with some cloud cover")
        elif result['cloud_coverage'] < 0.7:
            print("   ⛅ Broken cloud cover - Moderate visibility")
        else:
            print("   ☁️  Overcast conditions - Limited visibility")
        
        if result['cloud_density'] > 0.5:
            print("   🌧️  High cloud density suggests potential precipitation")
        elif result['cloud_density'] > 0.3:
            print("   💨 Moderate cloud density with variable conditions")
        else:
            print("   🌅 Low cloud density indicates stable weather")
        
        print(f"\n🎯 Visualization:")
        
        # Create high-quality visualization
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        fig.suptitle(f'Cloud Detection Results - {location_info["name"]}', fontsize=18, fontweight='bold')
        
        # Original satellite image with enhanced quality
        axes[0].imshow(img, interpolation='bilinear')
        axes[0].set_title(f"Sentinel-2 Satellite Image\n{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}", 
                         fontsize=12, fontweight='bold', pad=15)
        axes[0].axis('off')
        
        # Enhanced cloud mask visualization
        axes[1].imshow(mask, cmap='viridis', interpolation='bilinear')
        axes[1].set_title(f"Cloud Detection Mask\nCoverage: {result['cloud_coverage']:.1%}", 
                         fontsize=12, fontweight='bold', pad=15)
        axes[1].axis('off')
        
        # High-quality overlay visualization
        overlay = img.copy().astype(np.float32)
        # Ensure mask has the same shape as the image
        if mask.shape != img.shape[:2]:
            # Resize mask to match image dimensions with high quality
            from skimage.transform import resize
            mask_resized = resize(mask, img.shape[:2], order=1, preserve_range=True, anti_aliasing=True)
        else:
            mask_resized = mask
            
        # Create smooth cloud overlay
        cloud_areas = mask_resized > 0.3  # Lower threshold for better visualization
        cloud_intensity = np.clip(mask_resized * 0.7, 0, 1)  # Smooth intensity mapping
        
        # Apply red overlay with transparency
        overlay[cloud_areas, 0] = np.clip(overlay[cloud_areas, 0] * 0.3 + 255 * cloud_intensity[cloud_areas], 0, 255)
        overlay[cloud_areas, 1] = np.clip(overlay[cloud_areas, 1] * 0.3, 0, 255)
        overlay[cloud_areas, 2] = np.clip(overlay[cloud_areas, 2] * 0.3, 0, 255)
        
        axes[2].imshow(overlay.astype(np.uint8), interpolation='bilinear')
        axes[2].set_title(f"Cloud Overlay Analysis\nType: {result['cloud_type'].title()}", 
                         fontsize=12, fontweight='bold', pad=15)
        axes[2].axis('off')
        
        # Add colorbar for cloud mask
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = axes[1].imshow(mask, cmap='viridis')
        plt.colorbar(im, cax=cax, label='Cloud Probability')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.05, left=0.02, right=0.98, wspace=0.1, hspace=0.3)
        plt.show()
        
        print(f"\n📈 Summary:")
        print(f"   • Location: {location_info['name']}")
        print(f"   • Cloud Conditions: {result['cloud_type']}")
        print(f"   • Coverage Level: {result['cloud_coverage']:.1%}")
        print(f"   • Analysis Quality: {'High' if result['confidence'] > 0.7 else 'Moderate' if result['confidence'] > 0.5 else 'Low'}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 This can happen if:")
        print("   • No suitable images are found in GEE for the selected date range")
        print("   • The location has limited satellite coverage")
        print("   • Network connectivity issues with Google Earth Engine")
        print("   • Invalid coordinates or date format")
        
        if "No Sentinel-2 image found" in str(e):
            print("\n🔍 Troubleshooting:")
            print("   • Try a longer date range (e.g., 6 months to 1 year)")
            print("   • Check if the location has good satellite coverage")
            print("   • Verify the coordinates are correct")
            print("   • Try a different location with better data availability")

# Main execution
if __name__ == "__main__":
    run_cloud_detection_pipeline()






