"""
FAIR COMPARISON v2: Test ca 2 model P2-CIoU va P2-NWD
voi CUNG THAM SO de dam bao cong bang tuyet doi.
Luu output vao file txt rieng de dam bao khong mat du lieu.
"""
import subprocess
import re
import json
import os

YOLO_EXE = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\ai_env\Scripts\yolo.exe"
DATA_YAML = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\data_visdrone_only.yaml"
CWD = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI"

models = {
    "P2-CIoU": r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\runs\detect\runs\detect\ablation\yolo11m_p2_visdrone-4\weights\best.pt",
    "P2-NWD":  r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\runs\detect\ablation\yolo11m_p2_NWD_visdrone-3\weights\best.pt",
}

# THAM SO CHUNG - de YOLO dung mac dinh (conf=0.001, iou=0.7) giong het lan test P2 cu
COMMON_ARGS = [
    "imgsz=640",
    "batch=8",
    "device=0",
]

all_results = {}

for label, model_path in models.items():
    print("=" * 60)
    print(f"DANG TEST: {label}")
    print(f"Model: {model_path}")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"[LOI] Khong tim thay: {model_path}")
        continue

    cmd = [YOLO_EXE, "val", f"model={model_path}", f"data={DATA_YAML}"] + COMMON_ARGS

    # Gop stdout + stderr lai de khong mat output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore", cwd=CWD)
    out = result.stdout or ""

    # Luu raw output ra file de debug
    log_file = os.path.join(CWD, f"fair_test_{label.replace('-','_')}.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(out)
    print(f"[INFO] Raw output saved to: {log_file}")

    # Parse tung class - match ca tab va spaces
    class_data = {}
    for line in out.split('\n'):
        # Pattern: "                  all        548      21926      0.731      0.619      0.658       0.43"
        m = re.search(r'(all|Car|Motorcycle|Bus|Truck)\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', line)
        if m:
            cls_name = m.group(1).strip()
            p = float(m.group(2))
            r = float(m.group(3))
            map50 = float(m.group(4))
            map50_95 = float(m.group(5))
            class_data[cls_name] = {"P": p, "R": r, "mAP50": map50, "mAP50-95": map50_95}
            print(f"  {cls_name:<12}: P={p:.3f}  R={r:.3f}  mAP50={map50:.3f}  mAP50-95={map50_95:.3f}")

    all_results[label] = class_data
    print()

# In bang so sanh cuoi cung
print("\n" + "=" * 60)
print("BANG SO SANH CONG BANG TUYET DOI (CUNG THAM SO MAC DINH)")
print("=" * 60)
classes = ["all", "Car", "Motorcycle", "Bus", "Truck"]
for cls in classes:
    ciou = all_results.get("P2-CIoU", {}).get(cls, {})
    nwd = all_results.get("P2-NWD", {}).get(cls, {})
    delta_map50 = (nwd.get('mAP50',0) - ciou.get('mAP50',0)) * 100
    sign = "+" if delta_map50 >= 0 else ""
    print(f"{cls:<12} | P: {ciou.get('P',0)*100:.1f}% vs {nwd.get('P',0)*100:.1f}% | R: {ciou.get('R',0)*100:.1f}% vs {nwd.get('R',0)*100:.1f}% | mAP50: {ciou.get('mAP50',0)*100:.1f}% vs {nwd.get('mAP50',0)*100:.1f}% ({sign}{delta_map50:.1f}%)")

print("\n\nFINAL_JSON:" + json.dumps(all_results, indent=2))
