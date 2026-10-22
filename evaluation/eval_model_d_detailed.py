"""
Run Model D (Triple-Domain) evaluation with detailed per-class metrics
and confusion matrix extraction for all 3 datasets.
"""
from ultralytics import YOLO
import json, os
import numpy as np

if __name__ == '__main__':
    model = YOLO(r'E:\UAV-Traffic-Detection\AI-UAV-DL\AI\runs\detect\runs\benchmark\Model_D_TripleDomain\weights\best.pt')
    
    datasets = [
        ('VisDrone', 'data_visdrone_only.yaml'),
        ('Songdo', 'data_songdo_only.yaml'),
        ('JATAYU', 'data_jatayu_only.yaml'),
    ]
    
    all_results = {}
    
    for ds_name, yaml_file in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name}...")
        print(f"{'='*60}")
        
        res = model.val(data=yaml_file, split='val', workers=0, verbose=True)
        
        class_names = res.names
        
        ds_result = {
            'overall': {
                'precision': float(res.box.mp),
                'recall': float(res.box.mr),
                'mAP50': float(res.box.map50),
                'mAP50_95': float(res.box.map),
            },
            'per_class': {},
        }
        
        # Per-class metrics
        for i, cls_idx in enumerate(res.box.ap_class_index):
            name = class_names[cls_idx]
            ds_result['per_class'][name] = {
                'precision': float(res.box.p[i]),
                'recall': float(res.box.r[i]),
                'ap50': float(res.box.ap50[i]),
                'ap50_95': float(res.box.all_ap[i].mean()) if hasattr(res.box, 'all_ap') else float(res.box.ap[i]),
            }
        
        # Confusion matrix
        if hasattr(res, 'confusion_matrix'):
            cm = res.confusion_matrix
            if hasattr(cm, 'matrix'):
                matrix = cm.matrix.tolist()
                ds_result['confusion_matrix'] = matrix
                print(f"\nConfusion Matrix for {ds_name}:")
                # Print it nicely
                labels = [class_names[i] for i in range(len(class_names))] + ['BG/FP']
                print(f"{'':>15}", end='')
                for l in labels:
                    print(f"{l:>12}", end='')
                print()
                for i, row in enumerate(matrix):
                    lbl = labels[i] if i < len(labels) else f"cls_{i}"
                    print(f"{lbl:>15}", end='')
                    for val in row:
                        print(f"{val:>12.0f}", end='')
                    print()
        
        all_results[ds_name] = ds_result
    
    with open('model_d_detailed_eval.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All evaluations complete. Saved to model_d_detailed_eval.json")
