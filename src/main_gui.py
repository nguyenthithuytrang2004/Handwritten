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
import tkinter.font as tkfont

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


class MainGUI:
    """Main GUI application for HWR and OCR"""

    def __init__(self, root):
        self.root = root
        # Responsive UI: compute scale factor from screen resolution
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        scale_w = screen_w / 1366.0
        scale_h = screen_h / 768.0
        self.ui_scale = max(0.85, min(1.4, min(scale_w, scale_h)))

        self.root.title("Handwritten Character Recognition System")
        base_w, base_h = 640, 420
        self.root.geometry(f"{int(base_w * self.ui_scale)}x{int(base_h * self.ui_scale)}")
        self.root.minsize(int(480 * self.ui_scale), int(320 * self.ui_scale))
        self.root.resizable(True, True)

        # Apply a modern ttk theme and scale paddings
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        # Apply yellow / green accessible color theme (UI-only, no behavior change)
        try:
            bg = "#fff9e6"          # light warm yellow background
            accent = "#2e7d32"      # deep green for accents
            accent_light = "#a5d6a7" # lighter green for buttons
            text_dark = "#1b5e20"

            style.configure("TFrame", background=bg)
            style.configure("TLabel", background=bg, foreground=text_dark)
            style.configure("TButton", background=accent_light, foreground="black")
            style.map("TButton",
                      background=[('active', accent), ('!disabled', accent_light)],
                      foreground=[('active', 'white'), ('!disabled', 'black')])
            style.configure("Accent.TButton", background=accent, foreground="white")
            style.configure("TLabelframe", background=bg, foreground=text_dark)
            style.configure("TLabelframe.Label", background=bg, foreground=accent)
            # Status bar style
            style.configure("Status.TLabel", background="#f1f8e9", foreground=text_dark)
        except Exception:
            pass

        # Center window
        self.center_window()

        self.setup_ui()
        # Keep references to child windows to avoid garbage collection closing them
        self.child_windows = []

    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width() or 640
        height = self.root.winfo_height() or 420
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

        # Bind resize to adjust fonts dynamically for responsive layout
        self.root.bind("<Configure>", self.on_root_resize)

    def on_root_resize(self, event):
        """Adjust UI font sizes when the root window is resized."""
        try:
            # Only react to width changes (prevent loops)
            factor = event.width / max(1, int(640 * self.ui_scale))
            # Limit factor to reasonable range
            factor = max(0.7, min(1.6, factor))
            new_title = max(12, int(self.base_title_size * factor))
            self.title_font.configure(size=new_title)
            if self.default_font:
                new_default = max(9, int(self.base_default_size * factor))
                self.default_font.configure(size=new_default)
        except Exception:
            # swallow errors — resizing shouldn't break the app
            pass

    def setup_ui(self):
        """Setup the main UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="18")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        # Use a clearer, slightly larger font for headings scaled by ui_scale
        self.base_title_size = int(18 * self.ui_scale)
        self.title_font = tkfont.Font(family="Segoe UI", size=self.base_title_size, weight="bold")
        title_label = ttk.Label(main_frame, text="🖊️ Handwritten Character Recognition", font=self.title_font)
        title_label.pack(pady=(0, 8))

        subtitle_label = ttk.Label(
            main_frame,
            text="Chọn loại nhận diện bạn muốn sử dụng",
            font=("Segoe UI", int(10 * self.ui_scale))
        )
        subtitle_label.pack(pady=(0, 18))

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(expand=True)

        # HWR Button
        hwr_button = ttk.Button(
            buttons_frame,
            text="📝 Nhận diện chữ viết tay\n(Drawing Canvas)",
            command=self.launch_hwr,
            width=25,
            padding=(int(16 * self.ui_scale), int(12 * self.ui_scale))
        )
        hwr_button.pack(pady=(0, 12))

        hwr_desc = ttk.Label(
            buttons_frame,
            text="Vẽ chữ trên canvas và nhận diện ký tự",
            font=("Arial", 9),
            foreground="gray"
        )
        hwr_desc.pack(pady=(0, 12))

        # OCR Button
        ocr_button = ttk.Button(
            buttons_frame,
            text="📷 Nhận diện từ ảnh\n(OCR)",
            command=self.launch_ocr,
            width=25,
            padding=(int(16 * self.ui_scale), int(12 * self.ui_scale))
        )
        ocr_button.pack(pady=(0, 12))

        ocr_desc = ttk.Label(
            buttons_frame,
            text="Nhận diện text từ file ảnh",
            font=("Arial", 9),
            foreground="gray"
        )
        ocr_desc.pack(pady=(0, 8))

        # Footer
        footer_label = ttk.Label(
            main_frame,
            text="💡 Mẹo: Có thể chạy song song nhiều cửa sổ",
            font=("Arial", 8),
            foreground="blue"
        )
        # Footer and small controls row
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(int(10 * self.ui_scale), 0))
        footer_label.pack(in_=footer_frame, side=tk.LEFT)

        # Settings button (accessibility)
        settings_btn = ttk.Button(footer_frame, text="Settings", command=self.show_accessibility_settings)
        settings_btn.pack(side=tk.RIGHT, padx=(int(6 * self.ui_scale), 0))

        # Accessibility: Increase contrast toggle (simple)
        self.high_contrast_var = tk.BooleanVar(value=False)
        contrast_chk = ttk.Checkbutton(footer_frame, text="High contrast", variable=self.high_contrast_var, command=self.toggle_high_contrast)
        contrast_chk.pack(side=tk.RIGHT, padx=(int(6 * self.ui_scale), 0))
        # Track default font for UI scaling
        try:
            self.default_font = tkfont.nametofont("TkDefaultFont")
            self.base_default_size = max(10, int(self.default_font.cget("size") * self.ui_scale))
            self.default_font.configure(size=self.base_default_size)
        except Exception:
            self.default_font = None
            self.base_default_size = int(10 * self.ui_scale)

    def launch_hwr(self):
        """Launch Handwritten Character Recognition GUI"""
        try:
            # Create a child window (Toplevel) instead of a new Tk in another thread.
            from .hwr.gui import HWRApp
            top = tk.Toplevel(self.root)
            top.title("Handwritten Character Recognition")
            app = HWRApp(top)
            # keep reference so Python doesn't GC the wrapper (which can destroy widgets)
            self.child_windows.append(app)

            messagebox.showinfo(
                "Thông báo",
                "🖊️ Đã mở cửa sổ Nhận diện chữ viết tay.\n\n"
                "• Vẽ chữ trên canvas đen\n"
                "• Click 'Predict' để nhận diện\n"
                "• Click 'Save Sample' để thêm dữ liệu training"
            )

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động HWR:\n{str(e)}")

    def launch_ocr(self):
        """Launch OCR GUI"""
        try:
            # Create a child window (Toplevel) instead of a new Tk in another thread.
            from .ocr.gui import OCRApp
            top = tk.Toplevel(self.root)
            top.title("OCR Handwriting Recognition")
            app = OCRApp(top)
            self.child_windows.append(app)

            messagebox.showinfo(
                "Thông báo",
                "📷 Đã mở cửa sổ Nhận diện từ ảnh.\n\n"
                "• Click 'Chọn ảnh' để tải file\n"
                "• Chọn engine: Tesseract hoặc VietOCR\n"
                "• Click 'Chạy OCR' để nhận diện"
            )

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động OCR:\n{str(e)}")

    pass

    def show_accessibility_settings(self):
        """Show accessibility settings: font size, UI scale, high-contrast."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Accessibility Settings")
        dlg.geometry("480x220")
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Font size slider
        ttk.Label(frm, text="Font size for UI:").grid(row=0, column=0, sticky=tk.W)
        font_var = tk.IntVar(value=self.base_default_size)
        font_slider = ttk.Scale(frm, from_=8, to=20, orient=tk.HORIZONTAL, variable=font_var)
        font_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(8, 0))

        # Title font size slider
        ttk.Label(frm, text="Title font size:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        title_var = tk.IntVar(value=self.base_title_size)
        title_slider = ttk.Scale(frm, from_=12, to=28, orient=tk.HORIZONTAL, variable=title_var)
        title_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(8, 0), pady=(8, 0))

        # UI scale slider (Tk scaling)
        ttk.Label(frm, text="UI scale (affects widget sizing):").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(frm, from_=0.8, to=2.0, orient=tk.HORIZONTAL, variable=scale_var)
        scale_slider.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(8, 0), pady=(8, 0))

        # High contrast checkbox
        hc_var = tk.BooleanVar(value=self.high_contrast_var.get())
        hc_chk = ttk.Checkbutton(frm, text="High contrast", variable=hc_var)
        hc_chk.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        frm.columnconfigure(1, weight=1)

        def apply_and_close():
            # Apply font sizes
            try:
                new_default = int(font_var.get())
                if self.default_font:
                    self.default_font.configure(size=new_default)
                new_title = int(title_var.get())
                self.title_font.configure(size=new_title)
                # Apply UI scale via tk scaling
                try:
                    self.root.tk.call("tk", "scaling", float(scale_var.get()))
                except Exception:
                    pass
                # High contrast
                self.high_contrast_var.set(bool(hc_var.get()))
                self.toggle_high_contrast()
            except Exception:
                pass
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.pack(fill=tk.X, pady=(6, 8), padx=10)
        ttk.Button(btns, text="Apply", command=apply_and_close).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)

    def toggle_high_contrast(self):
        """Simple high-contrast mode toggler."""
        if self.high_contrast_var.get():
            style = ttk.Style()
            style.configure("TLabel", foreground="black", background="white")
            style.configure("TButton", foreground="black", background="#f0f0f0")
            self.root.configure(background="white")
        else:
            style = ttk.Style()
            style.configure("TLabel", foreground="black", background=None)
            style.configure("TButton", foreground="black", background=None)
            self.root.configure(background=None)


def run_main_gui():
    """Run the main GUI application"""
    # Start GUI (no blocking dependency checks; keep behavior unchanged)
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
