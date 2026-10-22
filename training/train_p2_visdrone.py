import os
import sys

# Fix encoding issue on Windows PowerShell
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from ultralytics import YOLO

def main():
    print("🚀 Bắt đầu huấn luyện YOLO11m với nhánh P2 trên tập VisDrone (Experiment E1) 🚀")
    
    # 1. Khởi tạo mô hình từ file cấu hình custom (yolo11m-p2.yaml)
    # Chúng ta sử dụng quy mô 'm' (Medium) bằng cách đặt tên file có hậu tố 'm'
    # ultralytics sẽ tự động lấy trọng số từ các lớp tương ứng và bỏ qua lớp Detect do cấu trúc thay đổi
    print("⏳ Đang build mô hình P2 (Scale Medium) và tải pretrained weights...")
    model_custom = YOLO("models/yolo11m-p2.yaml").load("yolo11m.pt")
    
    # 2. Bắt đầu train
    print("🔥 Bắt đầu quá trình Train...")
    results = model_custom.train(
        data="data_visdrone_only.yaml",
        epochs=100,
        patience=30,
        batch=8,
        imgsz=640,
        name="yolo11m_p2_visdrone",
        project="runs/detect/ablation", # Lưu kết quả vào thư mục ablation
        device=0, # Sử dụng GPU đầu tiên
        # Bật các augmentation cơ bản
        mosaic=1.0,
        mixup=0.1,
        amp=True, # Automatic Mixed Precision
    )
    
    print("🎉 Quá trình huấn luyện đã hoàn tất!")

if __name__ == "__main__":
    main()
