#!/usr/bin/env python3
"""
Google Earth Engine Setup Script
================================

This script helps set up Google Earth Engine integration for the solar energy models.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_gee_installation():
    """Check if GEE packages are installed"""
    try:
        import ee
        import geemap
        print("✅ GEE packages are installed")
        return True
    except ImportError as e:
        print(f"❌ GEE packages not found: {e}")
        return False

def setup_environment_variables():
    """Set up environment variables for GEE"""
    print("\n🔧 Setting up environment variables...")
    
    # Check if GOOGLE_APPLICATION_CREDENTIALS is set
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    if credentials_path:
        print(f"✅ GOOGLE_APPLICATION_CREDENTIALS already set: {credentials_path}")
        return True
    
    # Ask user for service account key path
    print("\nTo use Google Earth Engine, you need to set up authentication.")
    print("You have two options:")
    print("1. Use a service account key (recommended for production)")
    print("2. Use interactive authentication (for development)")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        key_path = input("Enter path to your service account key JSON file: ").strip()
        
        if not os.path.exists(key_path):
            print(f"❌ File not found: {key_path}")
            return False
        
        # Set environment variable
        if os.name == 'nt':  # Windows
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
            print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS={key_path}")
            print("Note: This is temporary. To make it permanent, add to your system environment variables.")
        else:  # Unix/Linux/Mac
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
            print(f"✅ Set GOOGLE_APPLICATION_CREDENTIALS={key_path}")
            print("Note: This is temporary. To make it permanent, add to your ~/.bashrc or ~/.zshrc:")
            print(f"export GOOGLE_APPLICATION_CREDENTIALS='{key_path}'")
        
        return True
    
    elif choice == "2":
        print("✅ Interactive authentication will be used when you run the models.")
        return True
    
    else:
        print("❌ Invalid choice")
        return False

def test_gee_connection():
    """Test GEE connection"""
    print("\n🔍 Testing GEE connection...")
    
    try:
        from gee_config import get_gee_config
        
        gee_config = get_gee_config()
        
        if gee_config.is_initialized:
            print("✅ GEE connection successful!")
            print(f"   Project ID: {gee_config.project_id}")
            return True
        else:
            print("❌ GEE connection failed")
            return False
            
    except Exception as e:
        print(f"❌ GEE connection test failed: {e}")
        return False

def create_sample_script():
    """Create a sample script to test the setup"""
    sample_script = """#!/usr/bin/env python3
\"\"\"
Sample GEE Usage Script
=======================

This script demonstrates basic usage of the GEE integration.
\"\"\"

from gee_config import get_gee_config
from datetime import datetime

def main():
    print("Testing GEE integration...")
    
    # Get GEE configuration
    gee_config = get_gee_config()
    
    if not gee_config.is_initialized:
        print("❌ GEE not initialized. Please check your setup.")
        return
    
    print("✅ GEE initialized successfully!")
    
    # Test coordinates (Colombo, Sri Lanka)
    lat, lon = 6.9271, 79.8612
    date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Fetch satellite data
        print(f"Fetching satellite data for ({lat}, {lon}) on {date}...")
        satellite_data = gee_config.fetch_satellite_image(lat, lon, date)
        
        print(f"✅ Data retrieved successfully!")
        print(f"   Cloud cover: {satellite_data['cloud_cover_percentage']:.1f}%")
        print(f"   Cloud type: {satellite_data['cloud_type']}")
        print(f"   Collection: {satellite_data['collection_type']}")
        
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("sample_gee_usage.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created sample_gee_usage.py")

def main():
    """Main setup function"""
    print("🚀 Google Earth Engine Setup for Solar Energy Models")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check GEE installation
    if not check_gee_installation():
        print("\n❌ Please install GEE packages first:")
        print("pip install earthengine-api geemap")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment_variables():
        print("\n❌ Environment setup failed")
        sys.exit(1)
    
    # Test connection
    if not test_gee_connection():
        print("\n❌ GEE connection failed. Please check your setup.")
        sys.exit(1)
    
    # Create sample script
    create_sample_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the test script: python test_gee_integration.py")
    print("2. Try the sample script: python sample_gee_usage.py")
    print("3. Start using the models with GEE integration!")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
