"""
BƯỚC 2: Train YOLO11m-P2 với NWD Loss
======================================
Script này sẽ:
  1. Kiểm tra NWD đã được patch chưa
  2. Train model P2 với NWD Loss (giữ nguyên 100% tham số như bản P2 gốc)
  3. Lưu kết quả vào thư mục ablation riêng

Chạy: python 02_train_nwd.py

CẤU HÌNH TRAINING (Giống hệt bản P2 gốc yolo11m_p2_visdrone-4):
  - Model:     yolo11m-p2.yaml (4 heads: P2, P3, P4, P5)
  - Data:      data_visdrone_only.yaml
  - Imgsz:     640
  - Epochs:    100
  - Patience:  30 (Early Stopping)
  - Batch:     8
  - Optimizer: auto
  - LR0:       0.01
  - Device:    GPU 0
"""

import os
import sys
import subprocess

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# === CẤU HÌNH ===
AI_DIR = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI"
YOLO_EXE = os.path.join(AI_DIR, "ai_env", "Scripts", "yolo.exe")
MODEL_YAML = os.path.join(AI_DIR, "models", "yolo11m-p2.yaml")
DATA_YAML = os.path.join(AI_DIR, "data_visdrone_only.yaml")
LOSS_FILE = os.path.join(AI_DIR, "ai_env", "Lib", "site-packages", "ultralytics", "utils", "loss.py")
PROJECT_DIR = os.path.join(AI_DIR, "runs", "detect", "ablation")
RUN_NAME = "yolo11m_p2_NWD_visdrone"

def check_nwd_patched():
    """Kiểm tra xem NWD đã được cấy vào loss.py chưa."""
    with open(LOSS_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    if "def nwd_loss(" not in content:
        print("[LỖI] NWD chưa được patch vào loss.py!")
        print("       Hãy chạy 'python 01_patch_nwd.py' trước.")
        sys.exit(1)
    if "ABLATION B3: NWD thay CIoU" not in content:
        print("[LỖI] BboxLoss chưa được sửa để dùng NWD!")
        print("       Hãy chạy 'python 01_patch_nwd.py' trước.")
        sys.exit(1)
    print("[OK] Xác nhận NWD đã được patch thành công vào loss.py")

def main():
    print("=" * 60)
    print("ABLATION B3 - BƯỚC 2: TRAIN YOLO11m-P2 VỚI NWD LOSS")
    print("=" * 60)

    # Kiểm tra các file cần thiết
    for path, name in [(YOLO_EXE, "yolo.exe"), (MODEL_YAML, "yolo11m-p2.yaml"), (DATA_YAML, "data_visdrone_only.yaml")]:
        if not os.path.exists(path):
            print(f"[LỖI] Không tìm thấy {name}: {path}")
            sys.exit(1)

    # Kiểm tra NWD đã patch chưa
    check_nwd_patched()

    # Tạo thư mục output
    os.makedirs(PROJECT_DIR, exist_ok=True)

    print("\n[INFO] Cấu hình Training:")
    print(f"  Model:    {MODEL_YAML}")
    print(f"  Data:     {DATA_YAML}")
    print(f"  Imgsz:    640")
    print(f"  Epochs:   100")
    print(f"  Patience: 30")
    print(f"  Batch:    8")
    print(f"  LR0:      0.01")
    print(f"  Output:   {PROJECT_DIR}/{RUN_NAME}")
    print()

    # === BẮT ĐẦU TRAIN ===
    cmd = [
        YOLO_EXE, "train",
        f"model={MODEL_YAML}",
        "pretrained=yolo11m.pt",  # Đảm bảo load backbone weights giống bản gốc
        f"data={DATA_YAML}",
        "imgsz=640",
        "epochs=100",
        "patience=30",
        "batch=8",
        "optimizer=auto",
        "lr0=0.01",
        "mixup=0.1",              # Khớp y hệt hyperparameter của bản P2 gốc
        "close_mosaic=10",
        "device=0",
        f"project={PROJECT_DIR}",
        f"name={RUN_NAME}",
        "exist_ok=False",
        "verbose=True",
    ]

    print("[START] Bắt đầu Training... (Ctrl+C để dừng an toàn)")
    print(f"[CMD] {' '.join(cmd)}")
    print("=" * 60)

    try:
        # Chạy trực tiếp, hiển thị output realtime
        process = subprocess.run(
            cmd,
            cwd=AI_DIR,
            encoding="utf-8",
            errors="ignore",
        )
        if process.returncode == 0:
            print("\n" + "=" * 60)
            print("[HOÀN TẤT] Training kết thúc thành công!")
            print(f"  Kết quả lưu tại: {PROJECT_DIR}/{RUN_NAME}")
            print("  Tiếp theo: chạy 'python 03_val_nwd.py' để đánh giá")
            print("=" * 60)
        else:
            print(f"\n[LỖI] Training kết thúc với mã lỗi: {process.returncode}")
            sys.exit(process.returncode)
    except KeyboardInterrupt:
        print("\n[DỪNG] Người dùng đã dừng Training.")
        print("  Model checkpoint gần nhất đã được tự động lưu.")
        print("  Có thể resume lại bằng cách dùng file last.pt")
        sys.exit(0)

if __name__ == "__main__":
    main()
