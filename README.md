# Handwritten Character Recognition System

Hệ thống nhận diện chữ viết tay (ký tự và OCR) — tài liệu này hướng dẫn các thành viên nhóm cách thiết lập môi trường, chuẩn bị dữ liệu và chạy training/finetune khi clone repo từ Git.

Mục tiêu: mọi thành viên trong team có thể clone repo, cài đặt dependencies, tải dữ liệu cần thiết và chạy model/GUI mà không bối rối.

## Mục lục
 - Giới thiệu nhanh
 - Yêu cầu & môi trường
 - Cài đặt nhanh
 - Dữ liệu (cách tải và chỗ đặt)
 - Chạy ứng dụng (GUI / CLI)
 - Training & Fine-tune (scripts)
 - Cấu trúc thư mục
 - Troubleshooting

## Giới thiệu nhanh
 - `src/hwr`: module nhận diện ký tự (model, preprocessing, training, GUI)
 - `src/ocr`: module nhận diện text từ ảnh (Tesseract, VietOCR)
 - `scripts/finetune_emnist.py`: script tải EMNIST và fine‑tune model với dữ liệu A_Z + user samples
 - `models/`: chứa model đã train

## Yêu cầu & môi trường
 - Python 3.8+ (tested on 3.8–3.11)
 - (Khuyến nghị) ảo hoá môi trường:
   - Windows / PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     ```
   - macOS / Linux:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
 - GPU: nếu có CUDA và tensorflow‑gpu, bạn có thể cài phiên bản tương ứng cho hiệu năng cao hơn.

## Cài đặt nhanh (Quickstart)

1) Clone repository và chuyển vào thư mục project:
```bash
git clone <repo-url>
cd Handwritten-main
```

2) Tạo virtual environment và cài dependencies (recommended):

- Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

- Windows (cmd.exe):
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

- macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Notes:
- Luôn chạy các lệnh từ project root (thư mục chứa `src/`) để tránh lỗi `ModuleNotFoundError: No module named 'src'`.
- Nếu gặp lỗi với `tensorflow` (phiên bản/pip wheel), báo cho tôi biết `python --version` và tôi sẽ cung cấp lệnh cài phù hợp (ví dụ CPU-only or CUDA-enabled wheels).

3) (OCR) Cài Tesseract binary (bắt buộc để dùng `pytesseract`):
- Windows (UB‑Mannheim build recommended): https://github.com/UB-Mannheim/tesseract/wiki
  - Mặc định cài vào `C:\Program Files\Tesseract-OCR`. Sau khi cài, kiểm tra:
    ```powershell
    tesseract --version
    ```
  - Nếu muốn cấu hình thủ công trong code, chỉnh `src/ocr/core.py`:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```
- macOS / Linux: cài bằng package manager (`brew install tesseract` hoặc `apt install tesseract-ocr`).

4) (Optional) If you prefer a single command to start the GUI from the project root after activating venv:
```powershell
python main.py gui
# or
python main.py --gui
```

5) Quick checks (after activation):
```powershell
python -c "import sys, pkgutil; print('python', sys.version)"
python -c "import numpy, PIL, tensorflow; print('ok')"
```

## Dữ liệu — tải & đặt đúng chỗ
Mục tiêu: mọi người dùng cùng một layout thư mục để training reproducible.

- `data/A_Z Handwritten Data.csv` (đã có trong repo) — NIST A_Z dataset (letters)
- `data/user_data/` — các sample do người dùng lưu từ GUI (structure: `data/user_data/<LABEL>/*.png`, ví dụ `data/user_data/A/123.png`).

Datasets bổ sung (khuyến nghị):
1. EMNIST (letters) — script tự tải giúp bạn:
   - Script: `scripts/finetune_emnist.py` sẽ tự download EMNIST bằng `tensorflow-datasets` và tiền xử lý, sau đó kết hợp với `data/A_Z Handwritten Data.csv` và `data/user_data/` (nếu có).
   - Chạy:
     ```bash
     pip install tensorflow-datasets
     python scripts/finetune_emnist.py --epochs 3 --batch-size 64
     ```
   - Kết quả: mô hình fine‑tuned sẽ được lưu ở `models/model_v2_emnist_finetuned.h5`.

2. (Tuỳ chọn) IAM Handwriting Database — nếu nhóm muốn từ character → word/sentence recognition. Cần đăng ký tải về và chuẩn hoá labels (phức tạp hơn).
   - Link: https://www.fki.inf.unibe.ch/databases/iam-handwriting-database/

Notes:
- Nếu cần thêm dataset từ Kaggle, tải file rồi đặt vào `data/` và viết script tiền xử lý tương ứng (mình có thể hỗ trợ).
- `scripts/finetune_emnist.py` đã xử lý rotate/flip EMNIST để alignment với NIST và convert labels phù hợp.

## Chạy ứng dụng
- Main GUI (gồm 2 nút: HWR và OCR):
  ```bash
  python main.py gui
  # or
  python main.py
  ```
- HWR GUI (chỉ canvas):
  ```bash
  python main.py hwr --gui
  ```
- OCR GUI:
  ```bash
  python main.py ocr --gui
  ```
- OCR CLI:
  ```bash
  python main.py ocr --image path/to/image.png --engine tesseract
  python main.py ocr --image path/to/image.png --engine vietocr
  ```

## Training & Fine‑tune
- Train từ đầu (sử dụng pipeline cũ):
  ```bash
  python scripts/train.py
  ```
- Fine‑tune kết hợp EMNIST + A_Z + user_data (được chuẩn hoá bởi script):
  ```bash
  pip install tensorflow-datasets
  python scripts/finetune_emnist.py --epochs 5 --batch-size 64
  ```
- Output: `models/model_v2_emnist_finetuned.h5`

## Cấu trúc dữ liệu mẫu (quan trọng cho team)
```
project_root/
├── data/
│   ├── A_Z Handwritten Data.csv
│   └── user_data/
│       ├── A/
│       │   └── 1768481301317_0.png
│       └── B/
├── models/
│   └── model_v2.h5
├── src/
└── scripts/
```

## Troubleshooting nhanh
- `ModuleNotFoundError: No module named 'src'`:
  - Chạy script từ project root (thư mục chứa `src/`), không chạy từ subfolder.
  - Hoặc đảm bảo `sys.path` chứa project root (scripts đã xử lý).

- `tesseract` không tìm thấy:
  - Kiểm tra `tesseract --version` sau khi cài.
  - Hoặc set trực tiếp trong `src/ocr/core.py`:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

- Nếu training quá chậm trên CPU: giảm batch size, giảm epochs, hoặc chạy trên máy có GPU.

## Gợi ý cho pull request / đóng góp
- Khi thêm dataset lớn, đừng commit data vào Git — thay vào đó lưu vào `data/` và mô tả link + script tải trong README.
- Nếu thêm pre-trained model lớn, đặt vào `models/` và ignore nếu cần, hoặc hướng dẫn tải từ release asset.

## Liên hệ nội bộ
- Nếu gặp lỗi khi chạy script training/finetune, gửi:
  - output terminal (log)
  - hệ điều hành và Python version
  - file `requirements.txt`

---
Tài liệu này được viết để mọi thành viên trong team có thể thiết lập môi trường và bắt đầu làm việc nhanh chóng. Nếu bạn muốn tôi bổ sung hướng dẫn cho Docker hoặc cho CI (GitHub Actions) thì nói mình sẽ thêm.

## 🏗️ Kiến trúc

### CNN Model
- **Input**: 28x28 grayscale images
- **Architecture**: Conv2D → BatchNorm → Conv2D → MaxPool → Dense
- **Classes**: 36 (0-9, A-Z)
- **Data**: MNIST digits + NIST letters + user data

### OCR Engines
- **Tesseract**: Fast, multi-language, rule-based
- **VietOCR**: Deep learning, better Vietnamese accuracy

## 📊 Độ chính xác

- **Character Recognition**: ~95% trên test set
- **OCR Accuracy**: Tùy thuộc vào chất lượng ảnh và engine
  - Tesseract: Good for clear text
  - VietOCR: Better for Vietnamese handwriting

## 🔧 Customization

### Thêm ngôn ngữ OCR
```python
# Trong src/ocr/core.py
ocr = TesseractOCR(lang="vie+eng+jpn")  # Thêm tiếng Nhật
```

### Thay đổi model architecture
```python
# Trong src/hwr/model.py
def build_model(num_classes):
    return tf.keras.Sequential([
        # Your custom architecture
    ])
```

### Thêm preprocessing steps
```python
# Trong src/ocr/core.py hoặc src/hwr/preprocessing.py
def custom_preprocess(image):
    # Your preprocessing logic
    return processed_image
```

## 🐛 Troubleshooting

### Model không load được
```bash
# Download pretrained model hoặc train từ đầu
python scripts/train.py
```

### OCR không hoạt động
```bash
# Check Tesseract installation
tesseract --version

# Install VietOCR
pip install vietocr torch torchvision
```

### GUI không hiển thị (Linux)
```bash
# Install tkinter
sudo apt-get install python3-tk
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [NIST Handwritten Characters](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [TensorFlow](https://www.tensorflow.org/)
