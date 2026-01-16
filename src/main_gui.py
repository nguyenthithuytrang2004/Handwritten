"""
Main GUI Application with two main options:
1. Handwritten Character Recognition (HWR)
2. Optical Character Recognition (OCR)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


class MainGUI:
    """Main GUI application for HWR and OCR"""

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Character Recognition System")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        # Center window
        self.center_window()

        self.setup_ui()

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

    def setup_ui(self):
        """Setup the main UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="🖊️ Handwritten Character Recognition",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(
            main_frame,
            text="Chọn loại nhận diện bạn muốn sử dụng",
            font=("Arial", 10)
        )
        subtitle_label.pack(pady=(0, 30))

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(expand=True)

        # HWR Button
        hwr_button = ttk.Button(
            buttons_frame,
            text="📝 Nhận diện chữ viết tay\n(Drawing Canvas)",
            command=self.launch_hwr,
            width=25,
            padding=(20, 15)
        )
        hwr_button.pack(pady=(0, 20))

        hwr_desc = ttk.Label(
            buttons_frame,
            text="Vẽ chữ trên canvas và nhận diện ký tự",
            font=("Arial", 9),
            foreground="gray"
        )
        hwr_desc.pack(pady=(0, 30))

        # OCR Button
        ocr_button = ttk.Button(
            buttons_frame,
            text="📷 Nhận diện từ ảnh\n(OCR)",
            command=self.launch_ocr,
            width=25,
            padding=(20, 15)
        )
        ocr_button.pack(pady=(0, 20))

        ocr_desc = ttk.Label(
            buttons_frame,
            text="Nhận diện text từ file ảnh",
            font=("Arial", 9),
            foreground="gray"
        )
        ocr_desc.pack(pady=(0, 30))

        # Footer
        footer_label = ttk.Label(
            main_frame,
            text="💡 Mẹo: Có thể chạy song song nhiều cửa sổ",
            font=("Arial", 8),
            foreground="blue"
        )
        footer_label.pack(pady=(10, 0))

    def launch_hwr(self):
        """Launch Handwritten Character Recognition GUI"""
        try:
            from .hwr.gui import run_gui

            # Run in separate thread to not block main GUI
            def run_hwr():
                try:
                    run_gui()
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể khởi động HWR GUI:\n{str(e)}")

            hwr_thread = threading.Thread(target=run_hwr, daemon=True)
            hwr_thread.start()

            messagebox.showinfo(
                "Thông báo",
                "🖊️ Đang mở cửa sổ Nhận diện chữ viết tay...\n\n"
                "• Vẽ chữ trên canvas đen\n"
                "• Click 'Predict' để nhận diện\n"
                "• Click 'Save Sample' để thêm dữ liệu training"
            )

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động HWR:\n{str(e)}")

    def launch_ocr(self):
        """Launch OCR GUI"""
        try:
            from .ocr.gui import run_gui

            # Run in separate thread to not block main GUI
            def run_ocr():
                try:
                    run_gui()
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể khởi động OCR GUI:\n{str(e)}")

            ocr_thread = threading.Thread(target=run_ocr, daemon=True)
            ocr_thread.start()

            messagebox.showinfo(
                "Thông báo",
                "📷 Đang mở cửa sổ Nhận diện từ ảnh...\n\n"
                "• Click 'Chọn ảnh' để tải file\n"
                "• Chọn engine: Tesseract hoặc VietOCR\n"
                "• Click 'Chạy OCR' để nhận diện"
            )

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động OCR:\n{str(e)}")


def run_main_gui():
    """Run the main GUI application"""
    root = tk.Tk()
    app = MainGUI(root)

    # Handle window close
    def on_closing():
        if messagebox.askokcancel("Thoát", "Bạn có muốn thoát ứng dụng?"):
            root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    run_main_gui()
