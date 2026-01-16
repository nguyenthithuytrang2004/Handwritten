#!/usr/bin/env python3
"""
Handwritten Character Recognition System
Main entry point for all functionality
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Handwritten Character Recognition System")
    parser.add_argument("mode", nargs="?", choices=["gui", "hwr", "ocr", "train"],
                       help="Mode: gui (main GUI), hwr (character recognition), ocr (text recognition), train (train model)")
    parser.add_argument("--gui", action="store_true", help="Run GUI mode for hwr/ocr")
    parser.add_argument("--image", help="Image path for OCR")
    parser.add_argument("--engine", choices=["tesseract", "vietocr"], default="tesseract",
                       help="OCR engine (default: tesseract)")
    parser.add_argument("--lang", default="vie+eng", help="OCR language (default: vie+eng)")

    args = parser.parse_args()

    # Handle main GUI mode
    if args.mode == "gui" or (not args.mode and args.gui):
        from src.main_gui import run_main_gui
        run_main_gui()
        return

    try:
        if args.mode == "hwr":
            if args.gui:
                from src.hwr.gui import run_gui
                run_gui()
            else:
                print("❌ HWR chỉ hỗ trợ GUI mode")
                print("💡 Sử dụng: python main.py hwr --gui")

        elif args.mode == "ocr":
            if args.gui:
                from src.ocr.gui import run_gui
                run_gui()
            else:
                if not args.image:
                    print("❌ Thiếu đường dẫn ảnh cho OCR CLI")
                    print("💡 Sử dụng: python main.py ocr --image path/to/image.png")
                    return

                from src.ocr.core import recognize_text
                if args.engine == "tesseract":
                    text = recognize_text(args.image, engine=args.engine, lang=args.lang)
                else:  # vietocr
                    text = recognize_text(args.image, engine=args.engine, device="cpu", beamsearch=True)
                print(text)

        elif args.mode == "train":
            from src.hwr.training import train_model
            print("🚀 Training model...")
            model, history = train_model()

            # Save model
            model.save("models/model_v2.h5")
            print("✅ Model saved to models/model_v2.h5")

            # Print results
            final_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            print(".2f")
            print(".2f")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Hãy cài đặt dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()