from ultralytics import YOLO
import os

def run_prediction():
    print("="*60)
    print("🎬 CHẠY NHẬN DIỆN VIDEO BẰNG YOLO11m MERGED 🎬")
    print("="*60)
    
    # Đường dẫn mô hình tốt nhất
    model_path = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\runs\detect\runs\train\yolo11m_merged\weights\best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: Không tìm thấy model tại {model_path}")
        return

    print("[*] Đang tải mô hình...")
    model = YOLO(model_path)
    
    video_paths = [r"E:\Downloads\220487.mp4"]
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy video {video_path}")
            continue
            
        print(f"\n[*] Đang nhận diện video: {video_path}")
        print("[*] Vui lòng chờ... quá trình này có thể mất vài phút tùy độ dài video.")

        # Chạy prediction
        results = model.predict(
            source=video_path,
            conf=0.25,        # Độ tự tin tối thiểu (lọc bớt nhiễu)
            save=True,        # Lưu video kết quả
            show=False,       # Không mở cửa sổ pop-up lúc đang chạy để tránh đơ máy
            line_width=2,     # Độ dày viền box (mỏng cho dễ nhìn xe nhỏ)
            device=0,         # Dùng Card RTX 4060
            project="runs/predict", 
            name="video_merged_out",
            exist_ok=True     # Ghi đè vào cùng 1 thư mục kết quả
        )
        
        # Lấy đường dẫn lưu video
        if len(results) > 0 and hasattr(results[0], 'save_dir'):
            print(f"✅ HOÀN TẤT VIDEO! Video kết quả lưu tại: {results[0].save_dir}")
    
    # Lấy đường dẫn lưu video
    if len(results) > 0 and hasattr(results[0], 'save_dir'):
        print("="*60)
        print(f"✅ HOÀN TẤT! Video kết quả đã được lưu tại:")
        print(f"   {results[0].save_dir}\\220453.mp4")
        print("="*60)

if __name__ == "__main__":
    run_prediction()
