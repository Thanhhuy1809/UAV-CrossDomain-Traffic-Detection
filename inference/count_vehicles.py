import cv2
import os
from ultralytics import YOLO

def main():
    print("="*60)
    print("🚀 ĐẾM XE VÀ VẼ GIAO DIỆN TÙY CHỈNH CHO 5 VIDEO 🚀")
    print("="*60)
    
    sequences = [
        "uav0000268_05773_v",
        "uav0000182_00000_v",
        "uav0000137_00458_v",
        "uav0000117_02622_v",
        "uav0000086_00000_v"
    ]
    
    model_path = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\runs\detect\runs\train\yolo11s_traffic_visdrone-2\weights\best.pt"
    print("[*] Đang tải mô hình...")
    model = YOLO(model_path)
    
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
    class_names = {0: 'O to', 1: 'Xe may', 2: 'Xe buyt', 3: 'Xe tai'}
    
    for seq_name in sequences:
        video_path = rf"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\raw_{seq_name}.mp4"
        output_path = rf"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\counted_{seq_name}.mp4"
        
        if not os.path.exists(video_path):
            continue
            
        print(f"\n[*] Đang xử lý đếm xe cho video: {seq_name}")
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        counted_ids = set()
        counts = {0: 0, 1: 0, 2: 0, 3: 0} 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            results = model.track(frame, persist=True, conf=0.25, iou=0.5, verbose=False, tracker="bytetrack.yaml")
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, cls in zip(boxes, ids, classes):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        counts[cls] += 1
                    
                    color = colors.get(cls, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (box[0], max(0, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
            # Draw UI
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (250, 170), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, "THONG KE LUU LUONG", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "-"*30, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset = 75
            for cls_id, name in class_names.items():
                cv2.putText(frame, f"{name}: {counts[cls_id]}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cls_id], 2)
                y_offset += 25
                
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"✅ Hoàn thành: counted_{seq_name}.mp4")

    print("\n" + "="*60)
    print("🎉 XONG TOÀN BỘ 5 VIDEO GIAO DIỆN XỊN!")
    print("="*60)

if __name__ == "__main__":
    main()
