"""
OCR GUI Application
Supports both Tesseract and VietOCR with modern interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
from PIL import Image, ImageTk, ImageOps, ImageFilter, ImageEnhance
import threading
import numpy as np

from .core import (
    OCREngine,
    TESSERACT_AVAILABLE,
    VIETOCR_AVAILABLE,
    check_image_quality,
    preprocess_for_tesseract,
    preprocess_for_vietocr,
    estimate_skew_angle,
)
from tkinter import ttk
from ..ui.widgets import RoundedLabelFrame

# Apply yellow/green theme for OCR window (UI-only)
try:
    style = ttk.Style()
    bg = "#fff9e6"
    text_dark = "#1b5e20"
    style.configure("OCR.TFrame", background=bg)
    style.configure("OCR.TLabel", background=bg, foreground=text_dark)
except Exception:
    pass


class OCRApp:
    """OCR GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("OCR Handwriting Recognition")
        # Responsive sizing
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        scale = max(0.8, min(1.4, min(screen_w / 1366.0, screen_h / 768.0)))
        self.ui_scale = scale
        self.root.geometry(f"{int(900 * scale)}x{int(700 * scale)}")
        self.root.resizable(True, True)

        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.vietocr_engine = None

        self.setup_ui()
        self.load_vietocr_model()

    def setup_ui(self):
        """Setup the GUI interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # File selection (rounded)
        file_frame = RoundedLabelFrame(main_frame, text="Chọn ảnh", padding=10)
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.inner.columnconfigure(1, weight=1)

        ttk.Button(file_frame.inner, text="Chọn ảnh...", command=self.select_image).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame.inner, text="Chưa chọn ảnh", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=tk.W)

        # Image preview (rounded)
        image_frame = RoundedLabelFrame(main_frame, text="Xem trước ảnh", padding=10)
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        image_frame.inner.columnconfigure(0, weight=1)
        image_frame.inner.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame.inner, text="Chưa có ảnh", anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Make preview responsive to container size
        try:
            image_frame.bind("<Configure>", lambda e: self._update_preview_resize())
        except Exception:
            pass

        # OCR configuration
        config_frame = RoundedLabelFrame(main_frame, text="Cấu hình OCR", padding=10)
        config_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        ttk.Label(config_frame.inner, text="Phương pháp OCR:").grid(row=0, column=0, sticky=tk.W, pady=5)

        self.ocr_method = tk.StringVar(value="tesseract")
        methods = []
        if TESSERACT_AVAILABLE:
            methods.append(("Tesseract OCR", "tesseract"))
        if VIETOCR_AVAILABLE:
            methods.append(("VietOCR (Chính xác hơn)", "vietocr"))

        if not methods:
            methods.append(("Không có OCR nào khả dụng", "none"))
            self.ocr_method.set("none")

        for text, value in methods:
            ttk.Radiobutton(config_frame.inner, text=text, variable=self.ocr_method,
                          value=value).grid(row=1, column=0, sticky=tk.W, pady=2)

        # Language selection (Tesseract only)
        ttk.Label(config_frame.inner, text="Ngôn ngữ:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.lang_var = tk.StringVar(value="vie+eng")
        lang_frame = ttk.Frame(config_frame.inner)
        lang_frame.grid(row=3, column=0, sticky=tk.W)

        ttk.Radiobutton(lang_frame, text="Tiếng Việt + Anh", variable=self.lang_var,
                       value="vie+eng").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(lang_frame, text="Chỉ tiếng Việt", variable=self.lang_var,
                       value="vie").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(lang_frame, text="Chỉ tiếng Anh", variable=self.lang_var,
                       value="eng").grid(row=2, column=0, sticky=tk.W)

        # Run button
        run_button = ttk.Button(config_frame.inner, text="🔍 Chạy OCR", command=self.run_ocr)
        run_button.grid(row=4, column=0, pady=(20, 0), sticky=(tk.W, tk.E))

        # Preprocessing options
        self.auto_preprocess_var = tk.BooleanVar(value=True)
        self.deskew_var = tk.BooleanVar(value=True)
        self.binarize_var = tk.BooleanVar(value=False)
        self.scale_var = tk.DoubleVar(value=2.5)

        ttk.Checkbutton(config_frame.inner, text="Tự động tiền xử lý (recommended)", variable=self.auto_preprocess_var).grid(row=5, column=0, sticky=tk.W, pady=(8, 2))
        ttk.Checkbutton(config_frame.inner, text="Deskew (tự động phát hiện nghiêng)", variable=self.deskew_var).grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(config_frame.inner, text="Binarize (chuyển ảnh nhị phân)", variable=self.binarize_var).grid(row=7, column=0, sticky=tk.W, pady=2)
        # Compare engines option
        self.compare_engines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame.inner, text="So sánh cả VietOCR và Tesseract để chọn kết quả tốt nhất", variable=self.compare_engines_var).grid(row=9, column=0, sticky=tk.W, pady=(8, 2))

        # Preview preprocessing button
        ttk.Button(config_frame.inner, text="🔎 Xem trước tiền xử lý", command=self.preview_preprocessing).grid(row=10, column=0, pady=(8, 0), sticky=(tk.W, tk.E))

        scale_frame = ttk.Frame(config_frame.inner)
        scale_frame.grid(row=8, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Label(scale_frame, text="Scale factor:").grid(row=0, column=0, sticky=tk.W)
        self.scale_slider = ttk.Scale(scale_frame, from_=1.0, to=4.0, orient=tk.HORIZONTAL, variable=self.scale_var)
        self.scale_slider.grid(row=0, column=1, padx=(6, 0))
        ttk.Label(scale_frame, textvariable=self.scale_var).grid(row=0, column=2, padx=(6, 0))

        # Results
        result_frame = RoundedLabelFrame(main_frame, text="Kết quả OCR", padding=10)
        result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.inner.columnconfigure(0, weight=1)
        result_frame.inner.rowconfigure(0, weight=1)

        font_size = int(11 * getattr(self, "ui_scale", 1.0))
        self.result_text = scrolledtext.ScrolledText(result_frame.inner, wrap=tk.WORD,
                                                     width=40, height=20, font=("Segoe UI", font_size))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Action buttons
        button_frame = ttk.Frame(result_frame.inner)
        button_frame.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))

        ttk.Button(button_frame, text="📋 Sao chép", command=self.copy_result).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="💾 Lưu kết quả", command=self.save_result).grid(row=0, column=1)

        # Status bar
        self.status_var = tk.StringVar(value="Sẵn sàng")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

    def load_vietocr_model(self):
        """Load VietOCR model in background"""
        if not VIETOCR_AVAILABLE:
            return

        def load_model():
            try:
                self.status_var.set("Đang tải VietOCR model...")
                self.vietocr_engine = OCREngine("vietocr", device="cpu", beamsearch=True)
                self.status_var.set("VietOCR model đã sẵn sàng!")
            except Exception as e:
                self.status_var.set(f"Lỗi tải VietOCR: {str(e)}")

        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()

    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Tất cả", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground="black")
            self.display_image_preview(file_path)

    def display_image_preview(self, image_path):
        """Display image preview"""
        try:
            img = Image.open(image_path)
            # copy image to decouple from file pointer and avoid resource issues
            img = img.copy()
            # store original for responsive rescaling
            self._orig_preview_image = img
            max_width, max_height = int(400 * getattr(self, "ui_scale", 1.0)), int(300 * getattr(self, "ui_scale", 1.0))
            thumb = img.copy()
            thumb.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(thumb)
            # Keep a reference both on the instance and the widget to prevent
            # the PhotoImage being garbage-collected which causes "pyimageX" errors.
            self.image_label.config(image=self.display_image, text="")
            self.image_label.image = self.display_image
        except Exception as e:
            # Ensure no stale image remains
            self.image_label.config(image=None, text=f"Lỗi hiển thị ảnh: {str(e)}")
            self.image_label.image = None

    def _update_preview_resize(self):
        """If an original preview image exists, rescale it to the current preview area."""
        try:
            if not getattr(self, "_orig_preview_image", None):
                return
            container = self.image_label.master
            w = max(50, container.winfo_width() - 20)
            h = max(50, container.winfo_height() - 20)
            img = self._orig_preview_image.copy()
            img.thumbnail((w, h), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.display_image, text="")
            self.image_label.image = self.display_image
        except Exception:
            pass

    def run_ocr(self):
        """Run OCR in background thread"""
        if not self.image_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return

        method = self.ocr_method.get()
        if method == "none":
            messagebox.showerror("Lỗi", "Không có phương pháp OCR nào khả dụng!")
            return
        # Pre-check image quality on main thread and decide preprocessing
        try:
            img = Image.open(self.image_path)
            diag = check_image_quality(img, engine=method)

            should_auto = self.auto_preprocess_var.get() and bool(diag.get("warnings"))
            # Also apply preprocessing if user explicitly toggled options
            user_requested = self.deskew_var.get() or self.binarize_var.get() or (float(self.scale_var.get()) != 2.5)

            if should_auto or user_requested:
                self.status_var.set("Áp dụng tiền xử lý theo cấu hình trước khi OCR...")
                ocr_input = self.apply_user_preprocessing(img, method)
            else:
                ocr_input = self.image_path
        except Exception:
            ocr_input = self.image_path

        def ocr_worker():
            try:
                self.status_var.set("Đang xử lý OCR...")
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Đang xử lý...\n")
                self.root.update()

                # If user enabled comparison and both engines available, run both and pick best
                def score_text(t):
                    if not t:
                        return 0.0
                    import re
                    # Count unicode letters as "good" characters; prefer longer texts
                    letters = len(re.findall(r"[^\W\d_]", t, flags=re.UNICODE))
                    total = max(len(t), 1)
                    # Score combines letter density and length (caps at 5 chars)
                    return (letters / total) * min(total / 5.0, 1.0)

                selected_text = None

                do_compare = self.compare_engines_var.get() and TESSERACT_AVAILABLE and VIETOCR_AVAILABLE
                results = {}

                if do_compare:
                    # Run VietOCR (uses loaded engine)
                    try:
                        if self.vietocr_engine is None:
                            raise Exception("VietOCR model chưa được tải! Vui lòng đợi...")
                        results["vietocr"] = self.vietocr_engine.recognize(ocr_input)
                    except Exception as e:
                        results["vietocr"] = ""

                    # Run Tesseract
                    try:
                        lang = self.lang_var.get()
                        tesser = OCREngine("tesseract", lang=lang)
                        results["tesseract"] = tesser.recognize(ocr_input)
                    except Exception:
                        results["tesseract"] = ""

                    # Score and pick best
                    best_engine = max(results.keys(), key=lambda k: score_text(results.get(k, "") or ""))
                    selected_text = results.get(best_engine, "")
                else:
                    if method == "tesseract":
                        lang = self.lang_var.get()
                        ocr = OCREngine("tesseract", lang=lang)
                        selected_text = ocr.recognize(ocr_input)
                    elif method == "vietocr":
                        if self.vietocr_engine is None:
                            raise Exception("VietOCR model chưa được tải! Vui lòng đợi...")
                        selected_text = self.vietocr_engine.recognize(ocr_input)
                    else:
                        selected_text = "Phương pháp không hợp lệ!"

                # Post-process: normalize whitespace
                if isinstance(selected_text, str):
                    sel = selected_text.strip()
                    sel = "\n".join([line.strip() for line in sel.splitlines() if line.strip()])
                else:
                    sel = ""

                # If we compared engines, show which engine was used in header
                header_method = method
                if do_compare:
                    # find which engine produced the selected_text
                    for k, v in results.items():
                        if v == selected_text:
                            header_method = f"compare->{k}"
                            break

                self.root.after(0, lambda: self.display_result(sel, header_method))

            except Exception as e:
                error_msg = f"Lỗi: {str(e)}"
                self.root.after(0, lambda: self.display_error(error_msg))

        thread = threading.Thread(target=ocr_worker, daemon=True)
        thread.start()
    def apply_user_preprocessing(self, pil_img, method_name):
        """Apply deskew, scale and optional binarization and return PIL image."""
        img_proc = pil_img
        try:
            # Deskew if requested
            if self.deskew_var.get():
                try:
                    arr_gray = np.array(img_proc.convert("L"))
                    angle = estimate_skew_angle(arr_gray)
                    if abs(angle) > 0.5:
                        img_proc = img_proc.rotate(-angle, expand=True, resample=Image.BICUBIC)
                except Exception:
                    pass

            scale_factor = float(self.scale_var.get())
            if method_name == "tesseract":
                img_proc = preprocess_for_tesseract(img_proc, scale_factor=scale_factor)
            else:
                img_proc = preprocess_for_vietocr(img_proc, scale_factor=scale_factor)

            if self.binarize_var.get():
                # Ensure grayscale then threshold
                gray = img_proc.convert("L")
                arr = np.array(gray)
                try:
                    # Otsu-like simple threshold
                    thresh = int(np.mean(arr) * 0.9)
                except Exception:
                    thresh = 128
                bin_arr = (arr > thresh) * 255
                img_proc = Image.fromarray(bin_arr.astype(np.uint8)).convert("L")
        except Exception:
            # fallback to original
            img_proc = pil_img
        return img_proc

    def preview_preprocessing(self):
        """Show a preview window with preprocessing applied to the selected image."""
        if not self.image_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        try:
            img = Image.open(self.image_path)
            processed = self.apply_user_preprocessing(img, self.ocr_method.get())
            # Resize preview for display
            max_w, max_h = 700, 500
            preview = processed.copy()
            preview.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            win = tk.Toplevel(self.root)
            win.title("Preview tiền xử lý")
            lbl = ttk.Label(win)
            lbl.pack(padx=10, pady=10)
            photo = ImageTk.PhotoImage(preview)
            lbl.config(image=photo)
            lbl.image = photo
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xem trước: {e}")

        # end preview_preprocessing

    def display_result(self, text, method):
        """Display OCR result"""
        self.result_text.delete(1.0, tk.END)

        header = f"=== KẾT QUẢ OCR ({method.upper()}) ===\n"
        header += "=" * 50 + "\n\n"
        self.result_text.insert(tk.END, header)
        self.result_text.insert(tk.END, text)

        if not text:
            self.result_text.insert(tk.END, "\n⚠️ Không tìm thấy text trong ảnh")

        self.status_var.set(f"Hoàn thành! (Phương pháp: {method})")

    def display_error(self, error_msg):
        """Display error message"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"❌ {error_msg}")
        self.status_var.set("Có lỗi xảy ra!")
        messagebox.showerror("Lỗi", error_msg)

    def copy_result(self):
        """Copy result to clipboard"""
        text = self.result_text.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Thành công", "Đã sao chép kết quả vào clipboard!")
        else:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để sao chép!")

    def save_result(self):
        """Save result to file"""
        text = self.result_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Cảnh báo", "Không có kết quả để lưu!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Thành công", f"Đã lưu kết quả vào:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể lưu file:\n{str(e)}")


def run_gui():
    """Run the OCR GUI application"""
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
