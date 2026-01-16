"""
Command Line Interface for OCR
"""

import sys
import os
from .core import recognize_text, TESSERACT_AVAILABLE, VIETOCR_AVAILABLE


def main():
    """
    Main CLI function for OCR
    Usage: python -m src.ocr.cli [image_path] [--engine tesseract|vietocr] [--lang lang]
    """
    if len(sys.argv) < 2:
        print("❌ Thiếu đường dẫn ảnh!")
        print("💡 Sử dụng: python -m src.ocr.cli <đường_dẫn_ảnh> [--engine tesseract|vietocr] [--lang vie+eng]")
        print("📚 Ví dụ: python -m src.ocr.cli image.png --engine vietocr")
        sys.exit(1)

    image_path = sys.argv[1]

    # Parse arguments
    engine = "tesseract"
    lang = "vie+eng"

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--engine" and i + 1 < len(args):
            engine = args[i + 1]
            i += 2
        elif args[i] == "--lang" and i + 1 < len(args):
            lang = args[i + 1]
            i += 2
        else:
            i += 1

    # Validate engine
    if engine not in ["tesseract", "vietocr"]:
        print(f"❌ Engine không hợp lệ: {engine}")
        print("💡 Engine hỗ trợ: tesseract, vietocr")
        sys.exit(1)

    # Check availability
    if engine == "tesseract" and not TESSERACT_AVAILABLE:
        print("❌ Tesseract không khả dụng!")
        print("💡 Cài đặt: pip install pytesseract")
        sys.exit(1)

    if engine == "vietocr" and not VIETOCR_AVAILABLE:
        print("❌ VietOCR không khả dụng!")
        print("💡 Cài đặt: pip install vietocr")
        sys.exit(1)

    # Check file
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        sys.exit(1)

    try:
        print(f"🔍 Đang xử lý ảnh: {image_path}...")
        print(f"⚙️  Engine: {engine}")

        if engine == "tesseract":
            text = recognize_text(image_path, engine="tesseract", lang=lang)
        else:
            text = recognize_text(image_path, engine="vietocr", device="cpu", beamsearch=True)

        if text:
            print("\n" + "="*60)
            print(f"KẾT QUẢ OCR ({engine.upper()}):")
            print("="*60)
            print(text)
            print("="*60 + "\n")
        else:
            print("⚠️ Không tìm thấy text trong ảnh")

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
