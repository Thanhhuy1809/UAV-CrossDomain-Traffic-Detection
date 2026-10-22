
"""
BƯỚC 1: Patch NWD Loss vào YOLO (Thay CIoU bằng NWD)
=====================================================
Script này sẽ:
  1. Backup file loss.py gốc (lưu thành loss.py.backup_original)
  2. Thêm hàm nwd_loss() vào file loss.py
  3. Sửa class BboxLoss để dùng NWD thay cho CIoU

Chạy: python 01_patch_nwd.py
"""

import os
import shutil
import sys

# === CẤU HÌNH ===
ULTRALYTICS_DIR = r"E:\UAV-Traffic-Detection\AI-UAV-DL\AI\ai_env\Lib\site-packages\ultralytics"
LOSS_FILE = os.path.join(ULTRALYTICS_DIR, "utils", "loss.py")
BACKUP_FILE = LOSS_FILE + ".backup_original"

def main():
    print("=" * 60)
    print("ABLATION B3 - BƯỚC 1: CẤY NWD LOSS VÀO YOLO")
    print("=" * 60)

    # --- Kiểm tra file tồn tại ---
    if not os.path.exists(LOSS_FILE):
        print(f"[LỖI] Không tìm thấy file: {LOSS_FILE}")
        sys.exit(1)

    # --- Backup file gốc (chỉ backup lần đầu) ---
    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(LOSS_FILE, BACKUP_FILE)
        print(f"[OK] Đã backup file gốc: {BACKUP_FILE}")
    else:
        print(f"[INFO] File backup đã tồn tại, bỏ qua backup.")
        # Khôi phục từ backup trước khi patch (đề phòng patch lặp)
        shutil.copy2(BACKUP_FILE, LOSS_FILE)
        print(f"[OK] Đã khôi phục loss.py từ backup trước khi patch lại.")

    # --- Đọc nội dung file loss.py ---
    with open(LOSS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # === PHẦN 1: Thêm hàm nwd_loss vào TRƯỚC class BboxLoss ===
    nwd_function = '''

# ==============================================================================
# NWD LOSS - Normalized Wasserstein Distance (Ablation B3)
# Paper: "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection"
# https://arxiv.org/abs/2110.13389
# ==============================================================================
def nwd_loss(pred_bboxes, target_bboxes, eps=1e-7):
    """
    Tính Normalized Wasserstein Distance giữa predicted và target bounding boxes.

    Ý tưởng cốt lõi:
    - Biến mỗi bounding box thành một phân phối Gaussian 2D
    - Tính khoảng cách Wasserstein giữa 2 phân phối đó
    - Chuẩn hóa (Normalize) để giá trị nằm trong [0, 1]

    Args:
        pred_bboxes: Predicted boxes [N, 4] dạng xyxy
        target_bboxes: Target boxes [N, 4] dạng xyxy
        eps: Hằng số tránh chia cho 0

    Returns:
        nwd: Tensor [N, 1] với giá trị trong [0, 1], càng gần 1 càng tốt
    """
    # Chuyển từ xyxy sang xywh (tâm x, tâm y, chiều rộng, chiều cao)
    # pred
    px1, py1, px2, py2 = pred_bboxes.chunk(4, dim=-1)
    pcx = (px1 + px2) / 2  # tâm x
    pcy = (py1 + py2) / 2  # tâm y
    pw = (px2 - px1).clamp(min=eps)  # chiều rộng
    ph = (py2 - py1).clamp(min=eps)  # chiều cao

    # target
    tx1, ty1, tx2, ty2 = target_bboxes.chunk(4, dim=-1)
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    # Tính Wasserstein Distance bậc 2 giữa 2 Gaussian 2D
    # W2 = (cx1 - cx2)^2 + (cy1 - cy2)^2
    #     + (w1/2 - w2/2)^2 + (h1/2 - h2/2)^2
    # (Với sigma = w/2 và h/2 cho phân phối Gaussian)
    dist_center = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
    dist_wh = (pw / 2 - tw / 2).pow(2) + (ph / 2 - th / 2).pow(2)
    wasserstein_2 = dist_center + dist_wh

    # Chuẩn hóa bằng hằng số C (dùng kích thước trung bình của target)
    # C tỷ lệ với kích thước vật thể, giúp NWD công bằng cho mọi scale
    C = (tw.pow(2) + th.pow(2)) / 2 + eps

    # Công thức NWD chuẩn: exp(-sqrt(W2) / C)
    # Giá trị NWD nằm trong (0, 1], càng gần 1 = càng khớp
    nwd = torch.exp(-torch.sqrt(wasserstein_2 + eps) / (C + eps))

    return nwd

'''

    # Chèn hàm nwd_loss TRƯỚC class BboxLoss
    marker = "class BboxLoss(nn.Module):"
    if "def nwd_loss(" in content:
        print("[CẢNH BÁO] Hàm nwd_loss đã tồn tại trong file. Patch lại...")
        # Đã được khôi phục từ backup ở trên, nên đọc lại
        with open(LOSS_FILE, "r", encoding="utf-8") as f:
            content = f.read()

    if marker not in content:
        print(f"[LỖI] Không tìm thấy marker '{marker}' trong loss.py")
        sys.exit(1)

    content = content.replace(marker, nwd_function + "\n" + marker)
    print("[OK] Đã thêm hàm nwd_loss() vào loss.py")

    # === PHẦN 2: Sửa class BboxLoss.forward() ===
    # Thay dòng:
    #   iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    # Bằng:
    #   iou = nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])

    old_line = "iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)"
    new_line = "iou = nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])  # ABLATION B3: NWD thay CIoU"

    if old_line not in content:
        print(f"[LỖI] Không tìm thấy dòng CIoU gốc trong BboxLoss.forward()")
        print("Có thể file đã bị patch trước đó. Hãy chạy 04_restore_original.py trước.")
        sys.exit(1)

    content = content.replace(old_line, new_line, 1)  # Chỉ thay 1 lần
    print("[OK] Đã thay CIoU bằng NWD trong BboxLoss.forward()")

    # === GHI FILE ===
    with open(LOSS_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print("\n" + "=" * 60)
    print("[HOÀN TẤT] Patch NWD thành công!")
    print(f"  - File đã sửa: {LOSS_FILE}")
    print(f"  - File backup:  {BACKUP_FILE}")
    print("  - Tiếp theo: chạy 'python 02_train_nwd.py' để bắt đầu Train")
    print("=" * 60)

if __name__ == "__main__":
    main()
