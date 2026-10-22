import os
import time
import cv2
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & THỜI GIAN
# ==========================================
MODEL_PATH = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\runs\detect\runs\benchmark\Model_D_TripleDomain\weights\best.pt"
VIDEO_PATH = r"E:\Downloads\E.People's Hall1.MP4"
SECONDS_TO_PROCESS = 60  # Chỉ chạy đúng 1 phút (60 giây) đầu tiên

def run_tracking_short(tracker_type="bytetrack.yaml"):
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy model tại: {MODEL_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"[LỖI] Không tìm thấy video tại: {VIDEO_PATH}")
        return

    print(f"\n{'='*60}")
    print(f" Khởi động Tracking: {tracker_type.upper()} ({SECONDS_TO_PROCESS} GIÂY ĐẦU TIÊN)")
    print(f"{'='*60}")
    
    # 1. Khởi tạo Model
    model = YOLO(MODEL_PATH)
    
    # 2. Mở Video gốc
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[LỖI] Không thể đọc được video.")
        return
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0: fps = 30 # Back-up nếu opencv không đọc được fps
    
    max_frames = fps * SECONDS_TO_PROCESS
    print(f"[INFO] Video: {width}x{height} @ {fps} FPS. Sẽ xử lý tối đa {max_frames} frames.")
    
    # 3. Chuẩn bị Video xuất ra (Lưu trong thư mục runs/tracking/)
    os.makedirs("runs/tracking", exist_ok=True)
    tracker_name = tracker_type.split('.')[0]
    out_path = f"runs/tracking/Model_D_{tracker_name}_30s_Test.mp4"
    
    # Dùng codec mp4v thông dụng cho mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    start_time = time.time()
    frame_count = 0
    
    # 4. Vòng lặp xử lý từng Frame
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break
        
        # Chạy tracking trên frame hiện tại (persist=True để giữ ID tracking qua từng frame)
        results = model.track(
            frame, 
            persist=True, 
            tracker=tracker_type, 
            conf=0.25, 
            iou=0.45, 
            verbose=False  # Tắt dòng log liên tục để đỡ rác màn hình
        )
        
        # Vẽ bounding box lên frame
        annotated_frame = results[0].plot(line_width=2)
        
        # Ghi frame đã vẽ vào video kết quả
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Đã xử lý {frame_count}/{max_frames} frames...")

    # 5. Quan trọng nhất: Đóng gói Video chuẩn xác
    cap.release()
    out.release()
    
    end_time = time.time()
    print(f"\n[THÀNH CÔNG] Đã lưu video chuẩn không bị hỏng tại: {os.path.abspath(out_path)}")
    print(f"Thời gian xử lý: {round(end_time - start_time, 2)} giây.")

if __name__ == "__main__":
    print("BẮT ĐẦU TEST TRACKING VỚI MODEL D (30 GIÂY)")
    
    # Chạy ByteTrack
    run_tracking_short("bytetrack.yaml")
    
    # Chạy BoT-SORT
    run_tracking_short("botsort.yaml")
    
    print("\n[HOÀN TẤT] Bạn có thể vào thư mục runs/tracking/ để mở xem ngay!")
