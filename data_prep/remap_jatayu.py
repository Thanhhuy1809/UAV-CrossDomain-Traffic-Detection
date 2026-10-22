import os
import glob

jatayu_path = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\JATAYU\Dataset1.v6-3x-aug-1280x720-gn_rk_kol_vb.yolov8"
label_files = glob.glob(os.path.join(jatayu_path, "**", "labels", "*.txt"), recursive=True)

# Bảng quy đổi (Map) các ID của JATAYU sang 4 class chuẩn của bạn
jat_map = {
    '0': '0', '3': '0', '5': '0', # Auto-rickshaw, Car, E-rickshaw -> Gom hết thành Car (0)
    '6': '1',                     # Motorcycle -> Đổi thành Motor (1)
    '2': '2',                     # Bus -> Giữ nguyên Bus (2)
    '8': '3', '9': '3'            # Tractor, Truck -> Gom thành Truck (3)
}
# Các ID không có trong bảng (Bicycle, Pedestrian) sẽ tự động bị xóa

count = 0
for f in label_files:
    new_lines = []
    with open(f, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if parts and parts[0] in jat_map:
                parts[0] = jat_map[parts[0]] # Ghi đè ID mới
                new_lines.append(" ".join(parts))
                
    # Ghi đè file cũ bằng nội dung mới (chỉ giữ lại 4 class xe)
    with open(f, 'w', encoding='utf-8') as file:
        file.write("\n".join(new_lines))
    count += 1

print(f"Đã xử lý xong {count} file nhãn của JATAYU!")
