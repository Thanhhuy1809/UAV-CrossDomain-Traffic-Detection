# Nhận diện Phương tiện Giao thông từ UAV trên Dữ liệu Đa miền (UAV-CrossDomain-Traffic-Detection)

Hệ thống nhận diện phương tiện giao thông từ camera UAV trên dữ liệu đa miền, xây dựng dựa trên kiến trúc YOLO11 kết hợp nhánh phát hiện độ phân giải cao P2 (Stride 4) và hàm mất mát Normalized Wasserstein Distance (NWD).

---

## 1. Giới thiệu tổng quan

Nhận diện phương tiện giao thông từ thiết bị bay không người lái (UAV) đối mặt với hai thách thức kỹ thuật lớn:
1. **Biến thiên kích thước cực lớn và vật thể siêu nhỏ (Tiny Objects):** Các phương tiện giao thông (đặc biệt là xe máy) thường chỉ chiếm diện tích rất nhỏ (dưới 16x16 pixel) khi chụp từ độ cao lớn, dẫn đến việc mất mát thông tin đặc trưng không gian qua các tầng tích chập sâu của mạng nơ-ron.
2. **Sự dịch chuyển phân phối dữ liệu (Domain Shift):** Mô hình được huấn luyện trên dữ liệu của một quốc gia thường bị suy giảm hiệu suất nghiêm trọng khi kiểm thử trên dữ liệu của quốc gia khác do sự khác biệt về góc nghiêng camera, độ cao bay, điều kiện chiếu sáng, mật độ giao thông và hình thái phương tiện đặc thù.

Dự án này cung cấp quy trình nghiên cứu, chuẩn hóa dữ liệu, huấn luyện và đánh giá trên 3 bộ dữ liệu UAV công khai đại diện cho 3 quốc gia (VisDrone - Trung Quốc, Songdo Vision - Hàn Quốc, JATAYU - Ấn Độ), đồng thời triển khai các cải tiến kiến trúc nhằm giải quyết điểm nghẽn nhận diện vật thể nhỏ.

---

## 2. Đóng góp kỹ thuật

- **Quy trình chuẩn hóa và đánh giá chéo đa miền (Cross-Domain Benchmark Protocol):** Đồng bộ hóa hệ thống nhãn về 4 phân lớp chuẩn (Car, Motorcycle, Bus, Truck) trên hơn 20.000 ảnh từ 3 quốc gia. Thiết lập 12 kịch bản đánh giá chéo giữa các mô hình đơn miền (Single-Domain) và mô hình đa miền (Triple-Domain).
- **Tích hợp nhánh phát hiện P2 (Stride 4):** Cải tiến kiến trúc YOLO11 nhằm khai thác bản đồ đặc trưng ở độ phân giải cao (P2/4), bảo toàn chi tiết biên dạng và vị trí của các phương tiện siêu nhỏ.
- **Tối ưu hàm mất mát với Normalized Wasserstein Distance (NWD):** Khắc phục nhược điểm của các hàm mất mát truyền thống dựa trên IoU (CIoU, GIoU) đối với vật thể nhỏ bằng cách mô hình hóa bounding box dưới dạng phân phối Gaussian 2 chiều và đo khoảng cách Wasserstein.

---

## 3. Kết quả thực nghiệm

### 3.1. Ma trận đánh giá chéo đa miền - mAP@0.5

| Mô hình huấn luyện \ Tập kiểm thử | VisDrone (Trung Quốc) | Songdo (Hàn Quốc) | JATAYU (Ấn Độ) |
| :--- | :---: | :---: | :---: |
| Model A (VisDrone) | 63.65% | 57.11% | 10.92% |
| Model B (Songdo) | 3.59% | 90.94% | 10.02% |
| Model C (JATAYU) | 2.44% | 33.13% | 70.78% |
| Model D (Triple-Domain) | 62.07% | 89.42% | 68.04% |

Nhận xét: Các mô hình đơn miền gặp hiện tượng sụp đổ độ chính xác khi kiểm thử chéo (ví dụ: Model B đạt 90.94% trên Songdo nhưng chỉ còn 3.59% trên VisDrone). Mô hình gộp Triple-Domain (Model D) duy trì độ chính xác cao và ổn định trên toàn bộ các miền mà không xảy ra hiện tượng lãng quên kiến thức cũ.

### 3.2. Ma trận độ chính xác vị trí - mAP@0.5:0.95

| Mô hình huấn luyện \ Tập kiểm thử | VisDrone (Trung Quốc) | Songdo (Hàn Quốc) | JATAYU (Ấn Độ) |
| :--- | :---: | :---: | :---: |
| Model A (VisDrone) | 42.03% | 36.09% | 6.89% |
| Model B (Songdo) | 1.73% | 65.67% | 5.64% |
| Model C (JATAYU) | 1.24% | 20.99% | 50.04% |
| Model D (Triple-Domain) | 40.50% | 63.77% | 47.30% |

### 3.3. Ma trận cân bằng - F1-Score

| Mô hình huấn luyện \ Tập kiểm thử | VisDrone (Trung Quốc) | Songdo (Hàn Quốc) | JATAYU (Ấn Độ) |
| :--- | :---: | :---: | :---: |
| Model A (VisDrone) | 65.15% | 59.94% | 20.08% |
| Model B (Songdo) | 9.35% | 88.81% | 20.78% |
| Model C (JATAYU) | 7.31% | 38.21% | 65.48% |
| Model D (Triple-Domain) | 63.91% | 87.06% | 65.65% |

---

## 4. Phân tích chi tiết hiệu suất Mô hình D theo từng lớp

| Tập kiểm thử | Phân lớp | Precision | Recall | F1-Score | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| VisDrone (Trung Quốc) | Car (Ô tô) | 79.72% | 80.05% | 79.89% | 82.44% | 58.17% |
| | Motorcycle (Xe máy) | 62.32% | 49.14% | 54.95% | 52.01% | 23.38% |
| | Bus (Xe buýt) | 75.64% | 55.38% | 63.94% | 60.28% | 43.75% |
| | Truck (Xe tải) | 67.47% | 47.08% | 55.46% | 53.54% | 36.71% |
| Songdo (Hàn Quốc) | Car (Ô tô) | 97.36% | 98.95% | 98.15% | 99.36% | 79.42% |
| | Motorcycle (Xe máy) | 75.49% | 60.49% | 67.16% | 67.36% | 25.91% |
| | Bus (Xe buýt) | 93.93% | 97.28% | 95.58% | 98.99% | 81.82% |
| | Truck (Xe tải) | 83.69% | 89.29% | 86.40% | 91.97% | 67.94% |
| JATAYU (Ấn Độ) | Car (Ô tô) | 47.82% | 78.12% | 59.32% | 67.32% | 43.59% |
| | Motorcycle (Xe máy) | 59.42% | 42.25% | 49.39% | 47.90% | 16.57% |
| | Bus (Xe buýt) | 56.35% | 67.58% | 61.45% | 62.57% | 51.84% |
| | Truck (Xe tải) | 87.24% | 87.55% | 87.39% | 94.35% | 77.20% |

Điểm nghẽn nhận diện: Lớp xe máy (Motorcycle) luôn có mức Recall thấp nhất trên cả 3 miền (chỉ đạt từ 42% đến 60%), đây là động lực chính cho việc đề xuất và phát triển nhánh P2 và hàm mất mát NWD trong giai đoạn cải tiến.

---

## 5. Cấu trúc thư mục dự án

`	ext
UAV-CrossDomain-Traffic-Detection/
├── configs/
│   ├── models/
│   │   └── yolo11m-p2.yaml          # File cấu hình kiến trúc YOLO11m tích hợp nhánh P2 (Stride 4)
│   ├── data_visdrone_only.yaml      # Cấu hình tập VisDrone
│   ├── data_songdo_only.yaml        # Cấu hình tập Songdo Vision
│   ├── data_jatayu_only.yaml        # Cấu hình tập JATAYU
│   └── data_triple_domain.yaml      # Cấu hình tập hợp nhất Triple-Domain
├── data_prep/                       # Tiền xử lý và đồng bộ hóa nhãn dữ liệu
│   ├── setup_visdrone.py            # Chuyển đổi nhãn VisDrone sang định dạng YOLO 4 lớp
│   ├── remap_songdo.py              # Ánh xạ lại ID nhãn Songdo Vision
│   ├── remap_jatayu.py              # Gom 10 lớp JATAYU về 4 lớp chuẩn
│   └── merge_datasets.py            # Hợp nhất dữ liệu thành tập Triple-Domain
├── training/                        # Huấn luyện mô hình
│   ├── train_benchmark.py           # Pipeline huấn luyện 4 mô hình và đánh giá chéo 12 lượt
│   ├── train_p2_visdrone.py         # Huấn luyện YOLO11m tích hợp nhánh P2 trên VisDrone
│   ├── patch_nwd_loss.py            # Tích hợp hàm mất mát NWD vào Ultralytics YOLO
│   └── train_nwd.py                 # Huấn luyện kết hợp P2 và NWD Loss
├── evaluation/                      # Đánh giá và phân tích chuyên sâu
│   ├── eval_model_d_detailed.py     # Đánh giá chi tiết từng lớp và ma trận nhầm lẫn
│   ├── analyze_motorcycle.py        # Phân tích phân bố kích thước pixel và nguyên nhân lỗi xe máy
│   ├── fair_compare_p2_nwd.py       # So sánh thực nghiệm công bằng P2-CIoU và P2-NWD
│   └── check_p2_val.py              # Kiểm tra nhanh checkpoint P2
├── inference/                       # Ứng dụng thực tế
│   ├── track_model_d.py             # Tracking phương tiện từ video UAV
│   ├── count_vehicles.py            # Đếm lưu lượng phương tiện qua vạch kẻ
│   └── predict_video.py             # Nhận diện bounding box trên video UAV
├── results/                         # Toàn bộ kết quả thực nghiệm gốc
│   ├── cross_domain_matrix.json     # Dữ liệu JSON gốc của ma trận đánh giá chéo
│   ├── cross_domain_matrix.csv      # Bảng CSV phục vụ báo cáo và bài báo
│   ├── model_d_detailed_eval.json   # Kết quả chi tiết theo từng lớp và ma trận nhầm lẫn
│   ├── motorcycle_analysis.json     # Thống kê phân bố pixel vật thể nhỏ
│   ├── fair_test_P2_CIoU.txt        # Kết quả đánh giá nhánh P2 với CIoU
│   └── fair_test_P2_NWD.txt         # Kết quả đánh giá nhánh P2 với NWD Loss
├── docs/                            # Tài liệu và báo cáo nghiên cứu
│   ├── Bao_Cao_Benchmark_CrossDomain.md # Báo cáo nghiên cứu đầy đủ bằng tiếng Việt
│   └── paper/mixstyle_domaingenarate.pdf # Tài liệu tham khảo MixStyle
├── weights/
│   └── README.md                    # Thông tin về các checkpoint mô hình
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md
`

---

## 6. Hướng dẫn sử dụng

### 6.1. Cài đặt môi trường
`ash
git clone https://github.com/Thanhhuy1809/UAV-CrossDomain-Traffic-Detection.git
cd UAV-CrossDomain-Traffic-Detection
pip install -r requirements.txt
`

### 6.2. Tiền xử lý dữ liệu
`ash
python data_prep/setup_visdrone.py
python data_prep/remap_songdo.py
python data_prep/remap_jatayu.py
python data_prep/merge_datasets.py
`

### 6.3. Huấn luyện mô hình Benchmark (Model A, B, C, D)
`ash
python training/train_benchmark.py
`

### 6.4. Huấn luyện YOLO11m với nhánh P2
`ash
python training/train_p2_visdrone.py
`

### 6.5. Huấn luyện với hàm mất mát NWD Loss
`ash
python training/patch_nwd_loss.py
python training/train_nwd.py
`

### 6.6. Chạy Tracking và Đếm phương tiện trên Video UAV
`ash
python inference/track_model_d.py --source /duong/dan/den/video.mp4 --conf 0.25
`

---

## 7. Giấy phép
Dự án được phân phối theo giấy phép MIT License.
