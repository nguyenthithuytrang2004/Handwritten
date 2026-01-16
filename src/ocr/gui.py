"""
OCR GUI Application
Supports both Tesseract and VietOCR with modern interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
from PIL import Image, ImageTk
import threading

from .core import OCREngine, TESSERACT_AVAILABLE, VIETOCR_AVAILABLE, check_image_quality, preprocess_for_tesseract, preprocess_for_vietocr


class OCRApp:
    """OCR GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("OCR Handwriting Recognition")
        self.root.geometry("900x700")
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

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Chọn ảnh", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Chọn ảnh...", command=self.select_image).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="Chưa chọn ảnh", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=tk.W)

        # Image preview
        image_frame = ttk.LabelFrame(main_frame, text="Xem trước ảnh", padding="10")
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, text="Chưa có ảnh", anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # OCR configuration
        config_frame = ttk.LabelFrame(main_frame, text="Cấu hình OCR", padding="10")
        config_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        ttk.Label(config_frame, text="Phương pháp OCR:").grid(row=0, column=0, sticky=tk.W, pady=5)

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
            ttk.Radiobutton(config_frame, text=text, variable=self.ocr_method,
                          value=value).grid(row=1, column=0, sticky=tk.W, pady=2)

        # Language selection (Tesseract only)
        ttk.Label(config_frame, text="Ngôn ngữ:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        self.lang_var = tk.StringVar(value="vie+eng")
        lang_frame = ttk.Frame(config_frame)
        lang_frame.grid(row=3, column=0, sticky=tk.W)

        ttk.Radiobutton(lang_frame, text="Tiếng Việt + Anh", variable=self.lang_var,
                       value="vie+eng").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(lang_frame, text="Chỉ tiếng Việt", variable=self.lang_var,
                       value="vie").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(lang_frame, text="Chỉ tiếng Anh", variable=self.lang_var,
                       value="eng").grid(row=2, column=0, sticky=tk.W)

        # Run button
        run_button = ttk.Button(config_frame, text="🔍 Chạy OCR", command=self.run_ocr)
        run_button.grid(row=4, column=0, pady=(20, 0), sticky=(tk.W, tk.E))

        # Results
        result_frame = ttk.LabelFrame(main_frame, text="Kết quả OCR", padding="10")
        result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD,
                                                     width=40, height=20, font=("Arial", 11))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Action buttons
        button_frame = ttk.Frame(result_frame)
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
            max_width, max_height = 400, 300
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.display_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.display_image, text="")
        except Exception as e:
            self.image_label.config(image="", text=f"Lỗi hiển thị ảnh: {str(e)}")

    def run_ocr(self):
        """Run OCR in background thread"""
        if not self.image_path:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return

        method = self.ocr_method.get()
        if method == "none":
            messagebox.showerror("Lỗi", "Không có phương pháp OCR nào khả dụng!")
            return
        # Pre-check image quality on main thread and offer auto-preprocessing
        try:
            img = Image.open(self.image_path)
            diag = check_image_quality(img, engine=method)
            if diag["warnings"]:
                # Auto-mode: apply preprocessing without asking
                if method == "tesseract":
                    processed = preprocess_for_tesseract(img)
                else:
                    processed = preprocess_for_vietocr(img)
                ocr_input = processed
                self.status_var.set("Áp dụng tiền xử lý tự động trước khi OCR...")
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

                if method == "tesseract":
                    lang = self.lang_var.get()
                    ocr = OCREngine("tesseract", lang=lang)
                    text = ocr.recognize(ocr_input)
                elif method == "vietocr":
                    if self.vietocr_engine is None:
                        raise Exception("VietOCR model chưa được tải! Vui lòng đợi...")
                    # VietOCR expects PIL image
                    if isinstance(ocr_input, (str, os.PathLike)):
                        text = self.vietocr_engine.recognize(ocr_input)
                    else:
                        text = self.vietocr_engine.predict(ocr_input) if hasattr(self.vietocr_engine, "predict") else self.vietocr_engine.predict(ocr_input)
                else:
                    text = "Phương pháp không hợp lệ!"

                self.root.after(0, lambda: self.display_result(text, method))

            except Exception as e:
                error_msg = f"Lỗi: {str(e)}"
                self.root.after(0, lambda: self.display_error(error_msg))

        thread = threading.Thread(target=ocr_worker, daemon=True)
        thread.start()

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
