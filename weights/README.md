# Thông tin Trọng số Mô hình (Pretrained Weights)

Do giới hạn dung lượng tệp tin của GitHub (tối đa 100 MB mỗi tệp tin đối với tài khoản tiêu chuẩn), các tệp tin trọng số mô hình (.pt) được quản lý và lưu trữ riêng biệt.

### Danh mục mô hình:
| Mô hình | Kiến trúc | Miền huấn luyện | Dung lượng | Tên tệp tin |
| :--- | :--- | :--- | :--- | :--- |
| Model A | YOLO11m (Nguyên bản) | VisDrone (Trung Quốc) | 38.6 MB | Model_A_VisDrone.pt |
| Model B | YOLO11m (Nguyên bản) | Songdo (Hàn Quốc) | 38.6 MB | Model_B_Songdo.pt |
| Model C | YOLO11m (Nguyên bản) | JATAYU (Ấn Độ) | 38.6 MB | Model_C_JATAYU.pt |
| Model D | YOLO11m (Nguyên bản) | Triple-Domain (Gộp 3 miền) | 38.6 MB | Model_D_TripleDomain.pt |
| YOLO11m-P2 | YOLO11m + Nhánh P2 (Stride 4) | VisDrone | 118.2 MB | yolo11m_p2_visdrone.pt |
| YOLO11m-P2-NWD | YOLO11m + P2 + NWD Loss | VisDrone | 118.2 MB | yolo11m_p2_nwd_visdrone.pt |

Lưu ý: Để sử dụng các trọng số trên trong mã nguồn, hãy đặt các tệp tin .pt tương ứng vào thư mục weights này.
