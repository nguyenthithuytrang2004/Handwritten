Handwritten Character Recognition — Quick Start (rút gọn)

Mục tiêu: cho mọi thành viên team biết nhanh cách cài, chạy và train lại model.

1) Chuẩn bị môi trường
- Yêu cầu: Python 3.8+
- Tạo virtualenv & cài dependencies:
  - Windows:
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

2) (OCR) Cài Tesseract (Windows)
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Kiểm tra: `tesseract --version`
- Nếu không muốn add PATH, sửa `src/ocr/core.py` để set `pytesseract.pytesseract.tesseract_cmd`.

3) Dữ liệu (folder `data/`)
- Có sẵn: `data/A_Z Handwritten Data.csv`
- User samples: `data/user_data/<LABEL>/*.png` (được lưu từ GUI)
- Tự động thêm EMNIST: `scripts/finetune_emnist.py` sẽ tải và tiền xử lý

4) Chạy ứng dụng
- Main GUI: `python main.py gui` (hoặc `python main.py`)
- HWR GUI: `python main.py hwr --gui`
- OCR GUI: `python main.py ocr --gui`
- OCR CLI: `python main.py ocr --image path/to/image.png --engine tesseract`

5) Fine‑tune model nhanh
- Chạy:
  ```bash
  pip install tensorflow-datasets
  python scripts/finetune_emnist.py --epochs 5 --batch-size 64
  ```
- Script lưu model fine‑tuned tại `models/model_v2_emnist_finetuned.h5` và cập nhật `models/model_v2.h5` để GUI dùng tự động.

6) Diagnostics
- Kiểm tra ảnh và lưu outputs: `python scripts/diagnose_sample.py --image path/to/image.png`
- Outputs nằm trong `diagnostic_out/`

7) Thêm Kaggle datasets (tuỳ chọn)
- Tạo API token trên Kaggle: https://www.kaggle.com/account → Create API Token → tải `kaggle.json`
- Đặt `kaggle.json` vào `%USERPROFILE%\.kaggle\kaggle.json` (Windows) hoặc `~/.kaggle/kaggle.json` (Linux/Mac)
- Mình sẽ tự download và tích hợp nếu bạn upload `kaggle.json` ở chat

8) Troubleshooting nhanh
- Chạy script từ project root để tránh lỗi import `src`
- Nếu training chậm, giảm batch size hoặc dùng GPU

Muốn mình sao chép nội dung này vào `README.md` thay thế bản hiện tại không? (gõ "ghi đè" để đồng ý)




