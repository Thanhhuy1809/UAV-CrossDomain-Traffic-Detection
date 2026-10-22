"""
==========================================================================
  MASTER TRAINING & EVALUATION PIPELINE
  Train 4 Models + Cross-Domain Evaluation Matrix
  YOLOv11m | 4 Classes: Car, Motorcycle, Bus, Truck
==========================================================================
  Model A: Train on VisDrone only
  Model B: Train on Songdo only
  Model C: Train on JATAYU only
  Model D: Train on Triple-Domain (VisDrone + Songdo + JATAYU)
==========================================================================
"""

from ultralytics import YOLO
import os
import csv
import json
import time
from datetime import timedelta

os.chdir(r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI")

# ===================== CAU HINH CHUNG =====================
BASE_MODEL   = "yolo11m.pt"
EPOCHS       = 100
IMGSZ        = 640
BATCH        = 8
DEVICE       = 0
WORKERS      = 4
PATIENCE     = 30
PROJECT      = "runs/benchmark"

# ===================== 4 MO HINH CAN TRAIN =====================
EXPERIMENTS = [
    {
        "name": "Model_A_VisDrone",
        "data": "data_visdrone_only.yaml",
        "desc": "Train ONLY on VisDrone2019"
    },
    {
        "name": "Model_B_Songdo",
        "data": "data_songdo_only.yaml",
        "desc": "Train ONLY on Songdo Vision"
    },
    {
        "name": "Model_C_JATAYU",
        "data": "data_jatayu_only.yaml",
        "desc": "Train ONLY on JATAYU (India UAV)"
    },
    {
        "name": "Model_D_TripleDomain",
        "data": "data_triple_domain.yaml",
        "desc": "Train on ALL 3 datasets (VisDrone + Songdo + JATAYU)"
    },
]

# ===================== 3 TAP TEST DE DANH GIA CHEO =====================
TEST_SETS = [
    {
        "name": "VisDrone",
        "data": "data_visdrone_only.yaml",
    },
    {
        "name": "Songdo",
        "data": "data_songdo_only.yaml",
    },
    {
        "name": "JATAYU",
        "data": "data_jatayu_only.yaml",
    },
]

# ===================== HAM TINH F1-SCORE =====================
def calc_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


# ===================== HAM IN KET QUA =====================
def print_metrics(tag, metrics):
    p = metrics.box.mp        # Mean Precision
    r = metrics.box.mr        # Mean Recall
    map50 = metrics.box.map50 # mAP@0.5
    map75 = metrics.box.map75 # mAP@0.75
    map5095 = metrics.box.map # mAP@0.5:0.95
    f1 = calc_f1(p, r)
    
    print(f"\n{'='*70}")
    print(f"  {tag}")
    print(f"{'='*70}")
    print(f"  Precision          : {p:.4f}  ({p*100:.2f}%)")
    print(f"  Recall             : {r:.4f}  ({r*100:.2f}%)")
    print(f"  F1-Score           : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  mAP@0.5            : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"  mAP@0.75           : {map75:.4f}  ({map75*100:.2f}%)")
    print(f"  mAP@0.5:0.95       : {map5095:.4f}  ({map5095*100:.2f}%)")
    
    # In ket qua tung class
    class_names = ['Car', 'Motorcycle', 'Bus', 'Truck']
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP50':>10} {'AP75':>10} {'AP50-95':>10}")
        print(f"  {'-'*75}")
        for i, name in enumerate(class_names):
            try:
                cp = metrics.box.p[i]
                cr = metrics.box.r[i]
                cf1 = calc_f1(cp, cr)
                cap50 = metrics.box.ap50[i]
                cap75 = metrics.box.ap75[i] if hasattr(metrics.box, 'ap75') else 0.0
                cap = metrics.box.ap[i] if hasattr(metrics.box, 'ap') else 0.0
                print(f"  {name:<15} {cp:>10.4f} {cr:>10.4f} {cf1:>10.4f} {cap50:>10.4f} {cap75:>10.4f} {cap:>10.4f}")
            except (IndexError, AttributeError):
                print(f"  {name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    print(f"{'='*70}")
    
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP75": round(map75, 4),
        "mAP50_95": round(map5095, 4),
    }


# ===================== MAIN PIPELINE =====================
def main():
    all_results = {}
    trained_models = {}
    
    total_start = time.time()
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#   MASTER PIPELINE: TRAIN 4 MODELS + CROSS-DOMAIN EVALUATION       #")
    print("#   Model: YOLOv11m | Classes: Car, Motorcycle, Bus, Truck           #")
    print("#   Benchmark: VisDrone + Songdo + JATAYU                            #")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    # ==========================================
    # PHASE 1: TRAIN 4 MODELS
    # ==========================================
    for idx, exp in enumerate(EXPERIMENTS, 1):
        # Kiem tra neu model nay da duoc train xong truoc do
        expected_dir = os.path.join("runs", "detect", PROJECT, exp["name"])
        expected_best = os.path.join(expected_dir, "weights", "best.pt")
        
        if os.path.exists(expected_best):
            print(f"\n[FOUND] Model {exp['name']} da train xong truoc do!")
            print(f"        Duong dan: {expected_best}")
            trained_models[exp["name"]] = expected_best
            continue
            
        print(f"\n{'*'*70}")
        print(f"  PHASE 1 - TRAINING [{idx}/4]: {exp['name']}")
        print(f"  {exp['desc']}")
        print(f"  Data: {exp['data']}")
        print(f"{'*'*70}\n")
        
        train_start = time.time()
        
        model = YOLO(BASE_MODEL)
        results = model.train(
            data=exp["data"],
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=PROJECT,
            name=exp["name"],
            patience=PATIENCE,
            save=True,
            save_period=10,
            plots=True,
            
            # Data Augmentation (Dong nhat cho ca 4 model de cong bang)
            mosaic=1.0,
            mixup=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )
        
        train_time = time.time() - train_start
        
        # Lay duong dan model tot nhat chinh xac tu results.save_dir
        best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
        trained_models[exp["name"]] = best_pt
        
        print(f"\n  [OK] {exp['name']} train xong trong {str(timedelta(seconds=int(train_time)))}")
        print(f"  [OK] Model saved: {best_pt}")
        
        # Danh gia lai bang model tot nhat de co metrics chinh xac
        val_best = YOLO(best_pt)
        val_res = val_best.val(
            data=exp["data"],
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            split="val",
            plots=False
        )
        print_metrics(f"VALIDATION RESULTS - {exp['name']}", val_res)
    
    # ==========================================
    # PHASE 2: CROSS-DOMAIN EVALUATION MATRIX
    # ==========================================
    print(f"\n\n{'#'*70}")
    print(f"#   PHASE 2: CROSS-DOMAIN EVALUATION MATRIX                         #")
    print(f"#   Danh gia cheo 4 Models x 3 Test Sets = 12 lan chay Val          #")
    print(f"{'#'*70}\n")
    
    matrix_results = {}
    
    for model_name, model_path in trained_models.items():
        matrix_results[model_name] = {}
        
        for test_set in TEST_SETS:
            tag = f"{model_name} --> Test on {test_set['name']}"
            print(f"\n  Evaluating: {tag} ...")
            
            model = YOLO(model_path)
            metrics = model.val(
                data=test_set["data"],
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                workers=WORKERS,
                split="val",  # Dung tap val de danh gia
                plots=False,
            )
            
            result = print_metrics(tag, metrics)
            matrix_results[model_name][test_set["name"]] = result
    
    # ==========================================
    # PHASE 3: IN BANG TONG KET (EVALUATION MATRIX)
    # ==========================================
    total_time = time.time() - total_start
    
    print(f"\n\n{'#'*70}")
    print(f"#   FINAL EVALUATION MATRIX                                         #")
    print(f"#   Tong thoi gian: {str(timedelta(seconds=int(total_time))):>48} #")
    print(f"{'#'*70}\n")
    
    # Bang mAP@0.5
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN mAP@0.5 MATRIX")
    print(f"{'='*70}")
    header = f"  {'Train \\\\ Test':<30} {'VisDrone':>12} {'Songdo':>12} {'JATAYU':>12}"
    print(header)
    print(f"  {'-'*66}")
    for model_name in trained_models:
        short_name = model_name.replace("Model_", "").replace("_", " ")
        row = f"  {short_name:<30}"
        for test_name in ["VisDrone", "Songdo", "JATAYU"]:
            val = matrix_results[model_name].get(test_name, {}).get("mAP50", 0)
            row += f" {val*100:>11.2f}%"
        print(row)
    print(f"{'='*70}")
    
    # Bang F1-Score
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN F1-SCORE MATRIX")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*66}")
    for model_name in trained_models:
        short_name = model_name.replace("Model_", "").replace("_", " ")
        row = f"  {short_name:<30}"
        for test_name in ["VisDrone", "Songdo", "JATAYU"]:
            val = matrix_results[model_name].get(test_name, {}).get("f1_score", 0)
            row += f" {val*100:>11.2f}%"
        print(row)
    print(f"{'='*70}")
    
    # Bang mAP@0.5:0.95
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN mAP@0.5:0.95 MATRIX")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*66}")
    for model_name in trained_models:
        short_name = model_name.replace("Model_", "").replace("_", " ")
        row = f"  {short_name:<30}"
        for test_name in ["VisDrone", "Songdo", "JATAYU"]:
            val = matrix_results[model_name].get(test_name, {}).get("mAP50_95", 0)
            row += f" {val*100:>11.2f}%"
        print(row)
    print(f"{'='*70}")
    
    # Bang Precision
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN PRECISION MATRIX")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*66}")
    for model_name in trained_models:
        short_name = model_name.replace("Model_", "").replace("_", " ")
        row = f"  {short_name:<30}"
        for test_name in ["VisDrone", "Songdo", "JATAYU"]:
            val = matrix_results[model_name].get(test_name, {}).get("precision", 0)
            row += f" {val*100:>11.2f}%"
        print(row)
    print(f"{'='*70}")
    
    # Bang Recall
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN RECALL MATRIX")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*66}")
    for model_name in trained_models:
        short_name = model_name.replace("Model_", "").replace("_", " ")
        row = f"  {short_name:<30}"
        for test_name in ["VisDrone", "Songdo", "JATAYU"]:
            val = matrix_results[model_name].get(test_name, {}).get("recall", 0)
            row += f" {val*100:>11.2f}%"
        print(row)
    print(f"{'='*70}")
    
    # ==========================================
    # PHASE 4: LUU KET QUA RA FILE JSON & CSV
    # ==========================================
    result_dir = os.path.join(PROJECT, "evaluation_results")
    os.makedirs(result_dir, exist_ok=True)
    
    # Luu JSON (day du chi tiet)
    json_path = os.path.join(result_dir, "cross_domain_matrix.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(matrix_results, f, indent=2, ensure_ascii=False)
    print(f"\n  [SAVED] Full results: {json_path}")
    
    # Luu CSV (de dan vao bai bao)
    csv_path = os.path.join(result_dir, "cross_domain_matrix.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["Model", "Test Dataset", "Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.75", "mAP@0.5:0.95"])
        for model_name, test_results in matrix_results.items():
            for test_name, vals in test_results.items():
                writer.writerow([
                    model_name, test_name,
                    vals["precision"], vals["recall"], vals["f1_score"],
                    vals["mAP50"], vals["mAP75"], vals["mAP50_95"]
                ])
    print(f"  [SAVED] CSV for paper: {csv_path}")
    
    print(f"\n{'#'*70}")
    print(f"#   ALL DONE! Tong thoi gian: {str(timedelta(seconds=int(total_time))):>39} #")
    print(f"#   4 Models trained, 12 Cross-domain evaluations completed.        #")
    print(f"#   Results saved to: {result_dir:<47} #")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
