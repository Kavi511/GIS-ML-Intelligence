#!/usr/bin/env python3
"""
Test script for improved cloud detection visualization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cloud_detection import (
    CloudDetectionModel, 
    CloudClassifier, 
    fetch_sentinel2_image,
    determine_climate_zone,
    display_location_insights,
    enhance_image_quality
)
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def test_cloud_detection_visualization():
    """Test the improved cloud detection with high-quality visualization"""
    
    print("☁️  TESTING IMPROVED CLOUD DETECTION VISUALIZATION")
    print("="*60)
    
    # Test coordinates (Los Angeles)
    test_coords = {
        "name": "Los Angeles, CA",
        "coords": (34.0549, -118.2426),
        "climate": "subtropical"
    }
    
    # Use today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📍 Test Location: {test_coords['name']}")
    lat, lon = test_coords['coords']
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    print(f"   📍 Geographic Position:")
    print(f"      • Latitude: {abs(lat):.4f}°{lat_dir} ({lat:.4f}°)")
    print(f"      • Longitude: {abs(lon):.4f}°{lon_dir} ({lon:.4f}°)")
    print(f"      • Hemispheres: {'Northern' if lat > 0 else 'Southern'}, {'Eastern' if lon > 0 else 'Western'}")
    print(f"📅 Date: {today}")
    
    try:
        # Display location insights
        display_location_insights(test_coords)
        
        # Initialize models
        print(f"\n🔧 Initializing cloud detection models...")
        cloud_model = CloudDetectionModel(model_path=None)
        classifier = CloudClassifier()
        
        # Fetch high-quality satellite image
        print(f"📡 Fetching high-quality Sentinel-2 satellite image...")
        img = fetch_sentinel2_image(
            lat=test_coords['coords'][0],
            lon=test_coords['coords'][1],
            start_date=today,
            end_date=today
        )
        
        print(f"✅ Image fetched successfully: {img.shape}")
        
        # Run cloud detection
        print(f"🔍 Running cloud detection analysis...")
        mask = cloud_model.predict(img)
        
        # Classify clouds
        result = classifier.classify_clouds(mask, img)
        
        # Display results
        print(f"\n📊 Cloud Detection Results:")
        print(f"   • Cloud Type: {result['cloud_type']}")
        print(f"   • Cloud Coverage: {result['cloud_coverage']:.1%}")
        print(f"   • Cloud Density: {result['cloud_density']:.3f}")
        print(f"   • Confidence: {result['confidence']:.1%}")
        
        # Create high-quality visualization
        print(f"\n🎯 Creating high-quality visualization...")
        create_enhanced_visualization(img, mask, result, test_coords)
        
        print(f"\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("💡 This might be due to:")
        print("   • No satellite data available for today")
        print("   • Network connectivity issues")
        print("   • Google Earth Engine authentication problems")

def create_enhanced_visualization(img, mask, result, location_info):
    """Create high-quality visualization with enhanced image processing"""
    
    # Set up high-quality plotting
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=150)
    fig.suptitle(f'High-Quality Cloud Detection Results - {location_info["name"]}', 
                fontsize=20, fontweight='bold', y=0.95)
    
    lat, lon = location_info['coords']
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    
    # Original satellite image with enhanced quality
    axes[0].imshow(img, interpolation='lanczos')
    axes[0].set_title(f"Enhanced Satellite Image\n{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}\n{lat_hemisphere}, {lon_hemisphere}", 
                     fontsize=14, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    # Enhanced cloud mask visualization with colorbar
    im1 = axes[1].imshow(mask, cmap='viridis', interpolation='lanczos')
    axes[1].set_title(f"Cloud Detection Mask\nCoverage: {result['cloud_coverage']:.1%}", 
                     fontsize=14, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    # Add colorbar for cloud mask
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_label('Cloud Probability', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # High-quality overlay visualization
    overlay = img.copy().astype(np.float32)
    
    # Ensure mask has the same shape as the image
    if mask.shape != img.shape[:2]:
        from skimage.transform import resize
        mask_resized = resize(mask, img.shape[:2], order=1, preserve_range=True, anti_aliasing=True)
    else:
        mask_resized = mask
    
    # Create smooth cloud overlay with transparency
    cloud_areas = mask_resized > 0.2  # Lower threshold for better visualization
    cloud_intensity = np.clip(mask_resized * 0.8, 0, 1)
    
    # Apply red overlay with smooth transparency
    overlay[cloud_areas, 0] = np.clip(overlay[cloud_areas, 0] * 0.2 + 255 * cloud_intensity[cloud_areas], 0, 255)
    overlay[cloud_areas, 1] = np.clip(overlay[cloud_areas, 1] * 0.2, 0, 255)
    overlay[cloud_areas, 2] = np.clip(overlay[cloud_areas, 2] * 0.2, 0, 255)
    
    axes[2].imshow(overlay.astype(np.uint8), interpolation='lanczos')
    axes[2].set_title(f"Cloud Overlay Analysis\nType: {result['cloud_type'].title()}", 
                     fontsize=14, fontweight='bold', pad=20)
    axes[2].axis('off')
    
    # Add analysis text box
    analysis_text = f"""
Cloud Analysis Results:
• Type: {result['cloud_type'].title()}
• Coverage: {result['cloud_coverage']:.1%}
• Density: {result['cloud_density']:.3f}
• Confidence: {result['confidence']:.1%}
• Quality: {'High' if result['confidence'] > 0.7 else 'Moderate' if result['confidence'] > 0.5 else 'Low'}
    """
    
    # Add text box with results
    fig.text(0.02, 0.02, analysis_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            verticalalignment='bottom')
    
    # Optimize layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.02, right=0.98, wspace=0.1, hspace=0.2)
    
    # Save high-quality image
    output_filename = f"cloud_detection_{location_info['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 High-quality visualization saved as: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    test_cloud_detection_visualization()
