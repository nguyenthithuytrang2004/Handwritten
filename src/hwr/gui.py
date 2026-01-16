"""
GUI for Handwritten Character Recognition
Drawing canvas to recognize handwritten characters
"""

import tkinter as tk
import tkinter.messagebox
import tkinter.simpledialog as simpledialog
import win32gui
import numpy as np
from PIL import Image, ImageGrab, ImageOps
from pathlib import Path
import time
import os

from .model import load_model, decode_prediction
from .preprocessing import segment_characters


class HWRApp:
    """Handwritten Character Recognition GUI Application"""

    def __init__(self, root):
        self.root = root
        self.root.geometry("400x400")
        self.root.title("Handwritten Character Recognition")

        # Load model
        try:
            self.model = load_model()
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.quit()
            return

        self.setup_ui()

    def setup_ui(self):
        """Setup the GUI interface"""
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, bg="black", height=300, width=300)
        self.canvas.pack(pady=10)
        self.canvas.bind('<B1-Motion>', self.mouse_event)

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        # Buttons
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Sample", command=self.save_sample).pack(side=tk.LEFT, padx=5)

    def mouse_event(self, event):
        """Handle mouse drawing event"""
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x, y, fill='white', outline='white', width=25)

    def capture_canvas_image(self):
        """Capture image from canvas"""
        canvas_handle = self.canvas.winfo_id()
        canvas_rect = win32gui.GetWindowRect(canvas_handle)
        return ImageGrab.grab(canvas_rect).convert("L")

    def clear(self):
        """Clear the canvas"""
        self.canvas.delete("all")

    def predict(self):
        """Predict characters from canvas"""
        img = self.capture_canvas_image()
        char_images, segments, _ = segment_characters(img)

        if not char_images:
            tkinter.messagebox.showinfo("Prediction", "No characters found.")
            return

        predictions = []
        for char_img in char_images:
            y_pred = self.model.predict(char_img, verbose=0)[0]
            predictions.append(decode_prediction(y_pred))

        # Calculate spacing for word formation
        widths = [end - start + 1 for start, end in segments]
        avg_width = float(np.mean(widths)) if widths else 0.0
        space_gap = max(int(avg_width * 0.6), 12)

        # Form words
        result = []
        for idx, pred in enumerate(predictions):
            result.append(pred)
            if idx < len(segments) - 1:
                gap = segments[idx + 1][0] - segments[idx][1]
                if gap > space_gap:
                    result.append(" ")

        tkinter.messagebox.showinfo("Prediction", "Prediction: " + "".join(result).strip())

    def save_sample(self):
        """Save drawn characters as training samples"""
        img = self.capture_canvas_image()
        _, _, char_arrays = segment_characters(img)

        if not char_arrays:
            tkinter.messagebox.showinfo("Save", "No characters found.")
            return

        prompt = "Found {} chars. Enter label string (spaces ignored):".format(len(char_arrays))
        label_text = simpledialog.askstring("Save samples", prompt)

        if not label_text:
            return

        CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        labels = [c for c in label_text.upper() if c != " "]

        if len(labels) != len(char_arrays):
            tkinter.messagebox.showinfo("Save", "Label length does not match detected characters.")
            return

        for label in labels:
            if label not in CLASSES:
                tkinter.messagebox.showinfo("Save", "Invalid label: {}".format(label))
                return

        timestamp = int(time.time() * 1000)

        for idx, (label, arr) in enumerate(zip(labels, char_arrays)):
            out_dir = Path("data/user_data") / label
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr.astype(np.uint8)).save(
                out_dir / "{}_{}.png".format(timestamp, idx)
            )

        tkinter.messagebox.showinfo("Save", "Saved {} samples.".format(len(char_arrays)))


def run_gui():
    """Run the HWR GUI application"""
    root = tk.Tk()
    app = HWRApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
