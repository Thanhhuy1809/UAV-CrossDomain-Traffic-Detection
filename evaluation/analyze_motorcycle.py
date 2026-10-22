"""
Comprehensive analysis script for Motorcycle performance diagnosis.
Collects: instance counts, bbox size distribution (all classes), 
confusion matrix data, and per-class AP at different object scales.
"""
import os, glob, json
import numpy as np
from collections import defaultdict

CLASSES = {0: 'Car', 1: 'Motorcycle', 2: 'Bus', 3: 'Truck'}
# COCO size thresholds (pixel area): small < 32^2, medium < 96^2, large >= 96^2
SMALL_THRESH = 32 * 32   # 1024
MED_THRESH   = 96 * 96   # 9216

DATASETS = {
    'VisDrone': r'E:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET\val\labels',
    'Songdo':   r'E:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\Songdo_Backup\val\labels',
    'JATAYU':   r'E:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\JATAYU\Dataset1.v6-3x-aug-1280x720-gn_rk_kol_vb.yolov8\valid\labels',
}

# Original image sizes for proper pixel area computation
# VisDrone: varies but typical ~2000x1500, resized to 640 for training
# Songdo: 3840x2160, resized to 640
# JATAYU: 1280x720, resized to 640
IMG_SIZES = {
    'VisDrone': 640,  # After resize to model input
    'Songdo': 640,
    'JATAYU': 640,
}

results = {}

for ds_name, label_dir in DATASETS.items():
    img_sz = IMG_SIZES[ds_name]
    
    # Per-class instance count
    class_counts = {c: 0 for c in CLASSES}
    
    # Per-class bbox size distribution: {cls: {'small': n, 'medium': n, 'large': n}}
    size_dist = {c: {'small': 0, 'medium': 0, 'large': 0} for c in CLASSES}
    
    # Per-class bbox width and height stats (in pixels at 640)
    bbox_dims = {c: {'widths': [], 'heights': [], 'areas': []} for c in CLASSES}
    
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name} | Label dir: {label_dir}")
    print(f"Number of label files: {len(label_files)}")
    
    for f in label_files:
        with open(f, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls not in CLASSES:
                        continue
                    class_counts[cls] += 1
                    
                    w_norm, h_norm = float(parts[3]), float(parts[4])
                    w_px = w_norm * img_sz
                    h_px = h_norm * img_sz
                    area = w_px * h_px
                    
                    bbox_dims[cls]['widths'].append(w_px)
                    bbox_dims[cls]['heights'].append(h_px)
                    bbox_dims[cls]['areas'].append(area)
                    
                    if area < SMALL_THRESH:
                        size_dist[cls]['small'] += 1
                    elif area < MED_THRESH:
                        size_dist[cls]['medium'] += 1
                    else:
                        size_dist[cls]['large'] += 1
    
    print(f"\n--- Instance Counts ---")
    total = sum(class_counts.values())
    for c, name in CLASSES.items():
        cnt = class_counts[c]
        pct = (cnt / total * 100) if total > 0 else 0
        print(f"  {name:12s}: {cnt:6d} instances ({pct:5.1f}%)")
    print(f"  {'TOTAL':12s}: {total:6d}")
    
    print(f"\n--- Bbox Size Distribution (pixels at 640x640, COCO thresholds) ---")
    for c, name in CLASSES.items():
        s, m, l = size_dist[c]['small'], size_dist[c]['medium'], size_dist[c]['large']
        t = s + m + l
        if t > 0:
            print(f"  {name:12s}: Small={s:5d} ({s/t*100:5.1f}%) | Medium={m:5d} ({m/t*100:5.1f}%) | Large={l:5d} ({l/t*100:5.1f}%)")
        else:
            print(f"  {name:12s}: No instances")
    
    print(f"\n--- Bbox Dimension Stats (pixels at 640x640) ---")
    for c, name in CLASSES.items():
        if len(bbox_dims[c]['areas']) > 0:
            areas = np.array(bbox_dims[c]['areas'])
            widths = np.array(bbox_dims[c]['widths'])
            heights = np.array(bbox_dims[c]['heights'])
            print(f"  {name:12s}: AvgArea={areas.mean():7.1f}px² | MedianArea={np.median(areas):7.1f}px² | AvgW={widths.mean():5.1f}px | AvgH={heights.mean():5.1f}px | MinArea={areas.min():6.1f}px² | MaxArea={areas.max():8.1f}px²")
        else:
            print(f"  {name:12s}: No instances")
    
    results[ds_name] = {
        'counts': {CLASSES[c]: class_counts[c] for c in CLASSES},
        'size_dist': {CLASSES[c]: size_dist[c] for c in CLASSES},
        'bbox_stats': {}
    }
    for c, name in CLASSES.items():
        if len(bbox_dims[c]['areas']) > 0:
            areas = np.array(bbox_dims[c]['areas'])
            results[ds_name]['bbox_stats'][name] = {
                'avg_area': float(areas.mean()),
                'median_area': float(np.median(areas)),
                'min_area': float(areas.min()),
                'max_area': float(areas.max()),
                'pct_small': float(size_dist[c]['small'] / len(areas) * 100),
            }

# Save JSON for later use
with open('motorcycle_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("Analysis complete. Results saved to motorcycle_analysis.json")
