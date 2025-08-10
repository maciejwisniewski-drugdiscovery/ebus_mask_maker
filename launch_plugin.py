#!/usr/bin/env python3
"""
Cross-platform launcher for napari-usg-masker plugin
"""

import sys
import os
import platform

def setup_environment():
    """Setup platform-specific environment variables"""
    if sys.platform == 'darwin':
        os.environ['QT_MAC_WANTS_LAYER'] = '1'
        print("🍎 Mac environment configured")
    elif sys.platform.startswith('win'):
        # Windows-specific setup if needed
        print("🪟 Windows environment configured")
    else:
        print("🐧 Linux environment configured")

def launch_napari_with_plugin():
    """Launch napari with the USG masker plugin"""
    try:
        # Setup environment first
        setup_environment()
        
        # Import napari
        import napari
        print("✅ Napari imported successfully")
        
        # Import the plugin widget
        from napari_usg_masker import USGMaskerWidget
        print("✅ USG Masker plugin imported successfully")
        
        # Create napari viewer
        print("🚀 Launching napari...")
        viewer = napari.Viewer()
        
        # Add the plugin widget
        widget = USGMaskerWidget(viewer)
        viewer.window.add_dock_widget(widget, name="USG Masker", area="right")
        
        print("✅ USG Masker widget added to napari")
        print("\n📖 Usage:")
        print("1. Use the 'Load USG Video' button to open your video")
        print("2. Use the masking tools to annotate frames")
        print("3. Use the new 'Export Frames & Masks (PNG+NPY)' button to export")
        
        # Run napari
        napari.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you've installed the plugin:")
        print("   pip install -e .")
        print("2. Check if napari is installed:")
        print("   napari --version")
        print("3. Try the installation script:")
        if sys.platform == 'darwin':
            print("   python install_mac.py")
        elif sys.platform.startswith('win'):
            print("   python install_windows.py")
        else:
            print("   pip install napari[all]")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching plugin: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🔬 Launching napari-usg-masker...")
    print("=" * 50)
    launch_napari_with_plugin()
