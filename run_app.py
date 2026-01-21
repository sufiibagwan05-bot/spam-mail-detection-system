"""
Quick launcher script for the Spam Email Detection System
Run this file to start the Streamlit application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not already installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import plotly
        print("✅ All required packages are already installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Installing required packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False

def main():
    """Main function to launch the application"""
    print("🛡️ Email Spam Detection System - Capstone Project")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project directory.")
        return
    
    # Install requirements if needed
    if not install_requirements():
        return
    
    print("\n🚀 Starting the Streamlit application...")
    print("📱 The app will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("\n💡 Tips:")
    print("   - Use Ctrl+C to stop the application")
    print("   - Refresh the browser if the app doesn't load immediately")
    print("   - Train the model first using the sidebar controls")
    print("\n" + "=" * 55)
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Thank you for using the Spam Detection System!")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("💡 Try running manually: streamlit run app.py")

if __name__ == "__main__":
    main()