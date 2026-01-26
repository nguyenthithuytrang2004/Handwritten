"""
Core OCR functionality supporting multiple engines
"""
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
import numpy as np
from PIL import Image
import os
import math

# Optional imports
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    VIETOCR_AVAILABLE = True
except ImportError:
    VIETOCR_AVAILABLE = False

# Configure Tesseract path for Windows
if os.name == "nt" and TESSERACT_AVAILABLE:
    tesseract_default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_default_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_default_path


def _ensure_cv2():
    """Raise a helpful error if OpenCV is not available."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is not installed. Install with: pip install opencv-python")


def preprocess_for_tesseract(img, scale_factor=2.5):
    """
    Preprocess image for Tesseract OCR

    Args:
        img: PIL Image
        scale_factor: Scale factor for image upsampling

    Returns:
        PIL.Image: Preprocessed image
    """
    _ensure_cv2()
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Cannot read image from: {img}")

    if isinstance(img, np.ndarray):
        # Convert to PIL if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Trim to content
    img = trim_to_content(img)

    # OpenCV processing
    arr = np.array(img)
    arr = cv2.resize(arr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    # Denoise
    arr = cv2.GaussianBlur(arr, (3, 3), 0)

    # Sharpen
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    arr = cv2.filter2D(arr, -1, kernel_sharp)

    # Binarize
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert
    thresh = 255 - thresh

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(thresh)


def preprocess_for_vietocr(img, scale_factor=2.5, visualize=False):
    """
    Preprocess image for VietOCR

    Args:
        img: PIL Image or path
        scale_factor: Scale factor for image upsampling
        visualize: If True, display intermediate steps using matplotlib (for debugging)

    Returns:
        PIL.Image: Preprocessed image
    """
    _ensure_cv2()
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Cannot read image from: {img}")
    else:
        # Convert PIL to OpenCV BGR if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if isinstance(img, np.ndarray):
        if len(img.shape) != 3 or img.shape[2] != 3:
            # Ensure color if grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Trim to content (converted to gray temporarily)
    gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pil_gray = Image.fromarray(gray_temp)
    trimmed = trim_to_content(pil_gray)
    img = np.array(trimmed)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Back to color

    if visualize:
        show("Original", img)

    # Resize (on color)
    img_resize = cv2.resize(
        img, None,
        fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )
    if visualize:
        show("Resize x2.5", img_resize)

    # Unsharp Mask (on color)
    blur = cv2.GaussianBlur(img_resize, (0, 0), 1.0)
    img_sharp = cv2.addWeighted(img_resize, 1.5, blur, -0.5, 0)
    if visualize:
        show("Sharpen", img_sharp)

    # Grayscale
    gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
    if visualize:
        show("Grayscale", gray, gray=True)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    if visualize:
        show("CLAHE", gray_clahe, gray=True)

    # Denoise (Blur)
    gray_blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
    if visualize:
        show("Blur", gray_blur, gray=True)

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    if visualize:
        show("Binary", binary, gray=True)

    # Convert to RGB for VietOCR
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


def estimate_skew_angle(arr_gray):
    """Estimate skew angle (degrees) of the text in the image using Hough lines fallback."""
    _ensure_cv2()
    edges = cv2.Canny(arr_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=100, minLineLength=30, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # normalize to -45..45
        if angle < -45:
            angle += 90
        if angle > 45:
            angle -= 90
        angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def check_image_quality(img, engine="tesseract"):
    """
    Check input image quality and return diagnostics.

    Args:
        img: PIL Image or path
        engine: 'tesseract' or 'vietocr' (affects suggestions)

    Returns:
        dict: keys: size, contrast, blur_var, skew, warnings (list)
    """
    _ensure_cv2()
    if isinstance(img, str):
        arr = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    elif isinstance(img, Image.Image):
        arr = np.array(img.convert("L"))
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            arr = img
    else:
        raise ValueError("Unsupported image type for quality check")

    h, w = arr.shape[:2]
    # contrast: std dev of intensities
    contrast = float(np.std(arr))
    # blur via Laplacian variance
    blur_var = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    # estimate skew
    try:
        skew = estimate_skew_angle(arr)
    except Exception:
        skew = 0.0

    warnings = []
    if w < 200 or h < 50:
        warnings.append("Image resolution is small; OCR accuracy may be low.")
    if contrast < 20:
        warnings.append("Low contrast detected; consider increasing contrast or brightness.")
    if blur_var < 50:
        warnings.append("Image may be blurry; try a sharper photo.")
    if abs(skew) > 5:
        warnings.append(f"Detected skew angle {skew:.1f}°; deskew recommended.")

    # small text heuristic: ratio of ink pixels
    ink_ratio = np.count_nonzero(arr < 200) / float(w * h)
    if ink_ratio < 0.02:
        warnings.append("Very little ink detected; text may be too small or faint.")

    return {
        "size": (w, h),
        "contrast": contrast,
        "blur_var": blur_var,
        "skew": skew,
        "ink_ratio": ink_ratio,
        "warnings": warnings,
    }


def trim_to_content(img_gray, threshold=20, pad=10):
    """
    Trim image to content area

    Args:
        img_gray: PIL grayscale image
        threshold: Binarization threshold
        pad: Padding around content

    Returns:
        PIL.Image: Trimmed image
    """
    arr = np.array(img_gray)
    mask = arr > threshold

    if not mask.any():
        return img_gray

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]

    height, width = arr.shape
    left = max(0, x0 - pad)
    upper = max(0, y0 - pad)
    right = min(width - 1, x1 + pad)
    lower = min(height - 1, y1 + pad)

    return img_gray.crop((left, upper, right + 1, lower + 1))


class TesseractOCR:
    """Tesseract OCR Engine"""

    def __init__(self, lang="vie+eng"):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not installed. Run: pip install pytesseract")
        self.lang = lang

    def recognize(self, image, config=None):
        """
        Recognize text from image using Tesseract

        Args:
            image: PIL Image or path
            config: Tesseract config string

        Returns:
            str: Recognized text
        """
        if config is None:
            config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"

        if isinstance(image, str):
            processed = preprocess_for_tesseract(image)
        else:
            processed = preprocess_for_tesseract(image)

        text = pytesseract.image_to_string(
            processed, lang=self.lang, config=config
        ).strip()

        return text


class VietOCR:
    """VietOCR Engine"""

    def __init__(self, device="cpu", beamsearch=True):
        if not VIETOCR_AVAILABLE:
            raise ImportError("VietOCR not installed. Run: pip install vietocr")

        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = beamsearch
        cfg["cnn"]["pretrained"] = True

        self.predictor = Predictor(cfg)

    def recognize(self, image):
        """
        Recognize text from image using VietOCR

        Args:
            image: PIL Image or path

        Returns:
            str: Recognized text
        """
        if isinstance(image, str):
            processed = preprocess_for_vietocr(image)
        else:
            processed = preprocess_for_vietocr(image)

        text = self.predictor.predict(processed).strip()
        return text


class OCREngine:
    """Unified OCR Engine supporting multiple backends"""

    def __init__(self, engine="tesseract", **kwargs):
        """
        Initialize OCR engine

        Args:
            engine: "tesseract" or "vietocr"
            **kwargs: Engine-specific parameters
        """
        self.engine = engine

        if engine == "tesseract":
            self.ocr = TesseractOCR(**kwargs)
        elif engine == "vietocr":
            self.ocr = VietOCR(**kwargs)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def recognize(self, image, **kwargs):
        """
        Recognize text from image

        Args:
            image: PIL Image or path to image
            **kwargs: Additional parameters for recognition

        Returns:
            str: Recognized text
        """
        return self.ocr.recognize(image, **kwargs)


# Convenience functions
def recognize_text(image, engine="tesseract", **kwargs):
    """
    Convenience function for text recognition

    Args:
        image: PIL Image or path
        engine: OCR engine to use
        **kwargs: Engine parameters

    Returns:
        str: Recognized text
    """
    ocr = OCREngine(engine, **kwargs)
    return ocr.recognize(image)
