"""
GUI for Handwritten Character Recognition
Drawing canvas to recognize handwritten characters
"""

import tkinter as tk
import tkinter.messagebox
import tkinter.simpledialog as simpledialog
import numpy as np
from PIL import Image, ImageGrab, ImageOps, ImageFilter, ImageDraw
from pathlib import Path
import time
import os
from tkinter import ttk
from ..ui.widgets import RoundedFrame

from .model import load_model, decode_prediction, CLASSES
from .preprocessing import segment_characters


class HWRApp:
    """Handwritten Character Recognition GUI Application"""

    def __init__(self, root):
        self.root = root
        # Responsive sizing based on screen resolution
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        scale = max(0.8, min(1.4, min(screen_w / 1366.0, screen_h / 768.0)))
        self.scale = scale
        width = int(420 * scale)
        height = int(420 * scale)
        self.root.geometry(f"{width}x{height}")
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
        # Canvas for drawing (size scales with screen)
        self.canvas_width = int(300 * self.scale)
        self.canvas_height = int(300 * self.scale)
        # Use a framed container so background theme shows through around the canvas
        container = tk.Frame(self.root, bg="#fff9e6")
        container.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(container, bg="black", height=self.canvas_height, width=self.canvas_width, highlightthickness=1, relief=tk.RIDGE)
        # Make canvas expand with window and be responsive
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind('<B1-Motion>', self.mouse_event)
        # Bind resize events to preserve and scale the internal drawing buffer
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Internal drawing buffer (PIL) to avoid OS ImageGrab/DPI issues.
        # White strokes on black background (L mode).
        self.draw_image = Image.new('L', (self.canvas_width, self.canvas_height), 0)
        self.draw_draw = ImageDraw.Draw(self.draw_image)

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        # Buttons
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Predict (Auto)", command=self.predict_auto_tune).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Sample", command=self.save_sample).pack(side=tk.LEFT, padx=5)

    def mouse_event(self, event):
        """Handle mouse drawing event"""
        x, y = event.x, event.y
        # Scale stroke radius with UI scale
        r = max(4, int(12 * self.scale))
        # Draw on canvas for live feedback
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white', width=0)
        # Draw on internal PIL buffer
        try:
            self.draw_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        except Exception:
            # If draw buffer missing, ignore
            pass

    def capture_canvas_image(self):
        """Capture image from canvas"""
        # Return a copy of the internal drawing buffer (already grayscale)
        try:
            return self.draw_image.copy()
        except Exception:
            # Fallback to screen grab if buffer missing
            self.root.update_idletasks()
            self.root.update()
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            bbox = (x, y, x + w, y + h)
            return ImageGrab.grab(bbox).convert("L")

    def clear(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        # recreate internal buffer
        self.draw_image = Image.new('L', (self.canvas_width, self.canvas_height), 0)
        self.draw_draw = ImageDraw.Draw(self.draw_image)

    def _on_canvas_resize(self, event):
        """Handle canvas size changes by resizing internal PIL buffer, attempting to preserve drawing."""
        try:
            new_w = max(10, event.width)
            new_h = max(10, event.height)
            if new_w == self.canvas_width and new_h == self.canvas_height:
                return
            # Try to scale existing drawing into new buffer
            try:
                old = self.draw_image
                resized = old.resize((new_w, new_h), Image.NEAREST)
                self.draw_image = resized
                self.draw_draw = ImageDraw.Draw(self.draw_image)
            except Exception:
                self.draw_image = Image.new('L', (new_w, new_h), 0)
                self.draw_draw = ImageDraw.Draw(self.draw_image)
            self.canvas_width = new_w
            self.canvas_height = new_h
        except Exception:
            pass

    def predict(self):
        """Predict characters from canvas"""
        img = self.capture_canvas_image()
        char_images, segments, char_arrays = segment_characters(img, threshold=127, pad=4)

        if not char_images:
            tkinter.messagebox.showinfo("Prediction", "No characters found.")
            return

        predictions = []
        debug_lines = []
        diagnostic_dir = Path("diagnostic_out")
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

        # Batch predict all characters for stability and speed
        try:
            batch = np.vstack(char_images)
            y_preds = self.model.predict(batch, verbose=0)
        except Exception:
            # Fallback to per-item predict if vstack fails
            y_preds = []
            for char_img in char_images:
                y_preds.append(self.model.predict(char_img, verbose=0)[0])
            y_preds = np.vstack(y_preds)

        # For each character, record top-3 and save diagnostic images
        for idx, y_pred in enumerate(y_preds):
            pred_char = decode_prediction(y_pred)
            predictions.append(pred_char)

            # top-3
            top_idx = np.argsort(-y_pred)[:3]
            probs = y_pred[top_idx]
            tops = [f"{CLASSES[i]}:{probs[k]:.2f}" for k, i in enumerate(top_idx)]
            debug_lines.append(f"Char {idx}: pred={pred_char}  top3={' '.join(tops)}")

            # save diagnostic images: input crop (upsampled) and 28x28
            try:
                arr28 = char_arrays[idx].astype(np.uint8)
                Image.fromarray(arr28).resize((200, 200), Image.NEAREST).save(diagnostic_dir / f"char_{idx}_input.png")
                Image.fromarray(arr28).save(diagnostic_dir / f"char_{idx}_28x28.png")
            except Exception:
                pass

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

    def predict_auto_tune(self):
        """Try several preprocessing configs and pick the prediction with highest model confidence."""
        img = self.capture_canvas_image()

        param_grid = {
            "thresholds": [100, 127, 150],
            "pads": [2, 4, 6],
            "dilates": [0, 1],
        }

        best_score = -1.0
        best_text = ""
        best_params = None

        for thr in param_grid["thresholds"]:
            for pad in param_grid["pads"]:
                for dil in param_grid["dilates"]:
                    try:
                        proc = img
                        # Apply simple dilation via MaxFilter to thicken strokes if requested
                        for _ in range(dil):
                            proc = proc.filter(ImageFilter.MaxFilter(3))

                        char_images, segments, _ = segment_characters(proc, threshold=thr, pad=pad)
                        if not char_images:
                            continue

                        try:
                            batch = np.vstack(char_images)
                            y_preds = self.model.predict(batch, verbose=0)
                        except Exception:
                            y_preds = []
                            for ci in char_images:
                                y_preds.append(self.model.predict(ci, verbose=0)[0])
                            y_preds = np.vstack(y_preds)

                        confidences = np.max(y_preds, axis=1)
                        avg_conf = float(confidences.mean()) if confidences.size else 0.0
                        # penalize if number of segments differs much from 1 (single-char assumption)
                        penalty = 1.0
                        if len(confidences) > 1:
                            penalty = 0.95

                        score = avg_conf * penalty
                        if score > best_score:
                            best_score = score
                            best_text = "".join(decode_prediction(y) for y in y_preds)
                            best_params = (thr, pad, dil)
                    except Exception:
                        continue

        if best_score < 0:
            tkinter.messagebox.showinfo("Prediction", "No characters found or prediction failed.")
            return

        tkinter.messagebox.showinfo("Prediction", f"Prediction (auto): {best_text}\nParams: thr={best_params[0]}, pad={best_params[1]}, dilate={best_params[2]}\nScore={best_score:.3f}")

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
