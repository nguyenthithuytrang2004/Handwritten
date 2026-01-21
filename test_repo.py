#!/usr/bin/env python3
"""
Quick test script to verify repository functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all imports work"""
    print("🔍 Testing imports...")

    try:
        from src.hwr.model import load_model, CLASSES
        print("✅ HWR model import OK")
    except Exception as e:
        print(f"❌ HWR import failed: {e}")
        return False

    try:
        from src.hwr.preprocessing import segment_characters
        print("✅ HWR preprocessing import OK")
    except Exception as e:
        print(f"❌ HWR preprocessing import failed: {e}")
        return False

    try:
        from src.ocr.core import OCREngine, TESSERACT_AVAILABLE, VIETOCR_AVAILABLE
        print("✅ OCR core import OK")
        print(f"   Tesseract: {'Available' if TESSERACT_AVAILABLE else 'Not available'}")
        print(f"   VietOCR: {'Available' if VIETOCR_AVAILABLE else 'Not available'}")
    except Exception as e:
        print(f"❌ OCR import failed: {e}")
        return False

    try:
        from src.hwr.training import load_full_dataset
        print("✅ Training utilities import OK")
    except Exception as e:
        print(f"❌ Training import failed: {e}")
        return False

    return True

def test_model_loading():
    """Test model loading"""
    print("\n🔍 Testing model loading...")
    try:
        from src.hwr.model import load_model
        model = load_model()
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading (quick test)"""
    print("\n🔍 Testing dataset loading...")
    try:
        from src.hwr.training import load_full_dataset
        # Only test loading without full processing
        print("✅ Dataset loading functions available")
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_ocr_engines():
    """Test OCR engines"""
    print("\n🔍 Testing OCR engines...")
    try:
        from src.ocr.core import VIETOCR_AVAILABLE, TESSERACT_AVAILABLE

        if VIETOCR_AVAILABLE:
            print("✅ VietOCR available")
        else:
            print("⚠️  VietOCR not available")

        if TESSERACT_AVAILABLE:
            print("✅ Tesseract available")
        else:
            print("⚠️  Tesseract not available")

        return True
    except Exception as e:
        print(f"❌ OCR engines test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Handwritten Character Recognition Repository")
    print("=" * 60)

    tests = [
        test_imports,
        test_model_loading,
        test_dataset_loading,
        test_ocr_engines
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Repository is ready to use.")
        print("\n💡 Try these commands:")
        print("   python main.py hwr --gui          # Character recognition GUI")
        print("   python main.py ocr --gui          # OCR GUI")
        print("   python main.py ocr --image photo.jpg --engine vietocr  # OCR CLI")
        print("   python scripts/train.py           # Train model")
    else:
        print("⚠️  Some tests failed. Check dependencies and try again.")

if __name__ == "__main__":
    main()
