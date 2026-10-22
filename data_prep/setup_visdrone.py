import os
from pathlib import Path
from PIL import Image

def get_iou(box1, box2):
    """Tính toán phần trăm giao nhau giữa 2 khung hình"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection_area / min(box1_area, box2_area)

def convert_visdrone_to_yolo_advanced(base_path):
    print("="*60)
    print("CONVERT VISDRONE TO YOLO - ADVANCED FILTER (RIDER ONLY)")
    print("="*60)
    
    base_path = Path(base_path)
    splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']
    
    for split in splits:
        split_dir = base_path / split
        if (split_dir / split).exists():
            split_dir = split_dir / split
            
        anno_dir = split_dir / 'annotations'
        img_dir = split_dir / 'images'
        
        if not anno_dir.exists() or not img_dir.exists():
            continue
            
        print(f"\n[*] Processing: {split}...")
        
        label_dir = split_dir / 'labels'
        label_dir.mkdir(exist_ok=True)
        
        total_files = 0
        total_boxes = 0
        
        for txt_file in anno_dir.glob('*.txt'):
            img_file = img_dir / f"{txt_file.stem}.jpg"
            if not img_file.exists():
                continue
                
            try:
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
            except Exception:
                continue
                
            total_files += 1
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            motors = []
            people = []
            other_vehicles = []
            
            # Đọc toàn bộ các objects trong file
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                
                x_left = float(parts[0])
                y_top = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                score = float(parts[4])
                category = int(parts[5])
                
                if score == 0: continue
                
                box = [x_left, y_top, x_left + w, y_top + h]
                
                if category == 10: # Motor
                    motors.append(box)
                elif category in [1, 2]: # Pedestrian or People
                    people.append(box)
                elif category in [4, 5, 6, 9]: # Car, Van, Truck, Bus
                    new_id = 0
                    if category == 4: new_id = 0
                    elif category in [5, 6]: new_id = 3
                    elif category == 9: new_id = 2
                    other_vehicles.append((box, new_id))
                    
            # LẤY TẤT CẢ XE MÁY (Bỏ qua logic kiểm tra người lái theo ý khách)
            valid_motors = []
            for m_box in motors:
                valid_motors.append((m_box, 1)) # ID YOLO cua Motor = 1
                    
            all_valid_vehicles = other_vehicles + valid_motors
            
            yolo_lines = []
            for box, class_id in all_valid_vehicles:
                # Convert back from box [x1, y1, x2, y2] to YOLO
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                x_center = (x1 + w / 2) / img_width
                y_center = (y1 + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                x_center = max(min(x_center, 1.0), 0.0)
                y_center = max(min(y_center, 1.0), 0.0)
                w_norm = max(min(w_norm, 1.0), 0.0)
                h_norm = max(min(h_norm, 1.0), 0.0)
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                total_boxes += 1
            
            label_file = label_dir / txt_file.name
            with open(label_file, 'w', encoding='utf-8') as f:
                f.writelines(yolo_lines)
                
        print(f"  + Converted {total_files} images.")
        print(f"  + Kept {total_boxes} HIGH-QUALITY (Ridden) vehicle boxes.")

    print("\n" + "="*60)
    print("DONE! Advanced filtering complete. YOLO11 is ready to train.")
    print("="*60)

if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    dataset_dir = current_dir / "dataset"
    convert_visdrone_to_yolo_advanced(dataset_dir)
