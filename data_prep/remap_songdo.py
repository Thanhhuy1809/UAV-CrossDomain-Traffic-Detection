"""
Remap nhãn (class ID) của Songdo Vision cho khớp với VisDrone.

Songdo gốc:    0=Car, 1=Bus,        2=Truck, 3=Motorcycle
VisDrone (ta):  0=Car, 1=Motorcycle, 2=Bus,   3=Truck

Mapping: Songdo 0 -> 0 (Car giữ nguyên)
         Songdo 1 -> 2 (Bus -> id 2)
         Songdo 2 -> 3 (Truck -> id 3)
         Songdo 3 -> 1 (Motorcycle -> id 1)
"""
import os
import glob

REMAP = {0: 0, 1: 2, 2: 3, 3: 1}

def remap_labels(label_dir):
    txt_files = glob.glob(os.path.join(label_dir, "songdo_*.txt"))
    total = 0
    remapped = 0
    
    for fpath in txt_files:
        new_lines = []
        with open(fpath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_id = int(parts[0])
            if old_id in REMAP:
                parts[0] = str(REMAP[old_id])
                if old_id != REMAP[old_id]:
                    remapped += 1
            new_lines.append(" ".join(parts) + "\n")
            total += 1
        
        with open(fpath, 'w') as f:
            f.writelines(new_lines)
    
    print(f"  Scanned {len(txt_files)} files, {total} boxes total, {remapped} boxes remapped.")

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 REMAP SONGDO LABELS -> VISDRONE CLASS ORDER")
    print("=" * 60)
    
    train_labels = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-train\labels"
    val_labels = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-val\labels"
    
    print("\n[*] Remapping TRAIN labels...")
    remap_labels(train_labels)
    
    print("[*] Remapping VAL labels...")
    remap_labels(val_labels)
    
    print("\n" + "=" * 60)
    print("✅ REMAP HOÀN TẤT! Nhãn Songdo giờ đã khớp với VisDrone.")
    print("=" * 60)
