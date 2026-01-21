#!/usr/bin/env python3
"""
Demo script showing how to use the main GUI
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_main_gui():
    """Demo the main GUI"""
    print("🚀 Launching Main GUI...")
    print("This will show a window with 2 big buttons:")
    print("1. 📝 Nhận diện chữ viết tay (Drawing Canvas)")
    print("2. 📷 Nhận diện từ ảnh (OCR)")
    print("\nTo run manually: python main.py gui")

    from src.main_gui import run_main_gui
    run_main_gui()

if __name__ == "__main__":
    demo_main_gui()
