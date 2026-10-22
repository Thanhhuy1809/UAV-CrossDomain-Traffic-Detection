# Pretrained Model Weights

Due to GitHub's file size limits (especially models exceeding 100 MB), model weights are managed separately.

### Model Catalog:
| Model | Architecture | Training Domain | Size | File Name |
| :--- | :--- | :--- | :--- | :--- |
| **Model A** | YOLO11m (Baseline) | VisDrone (China) | 38.6 MB | Model_A_VisDrone.pt |
| **Model B** | YOLO11m (Baseline) | Songdo (Korea) | 38.6 MB | Model_B_Songdo.pt |
| **Model C** | YOLO11m (Baseline) | JATAYU (India) | 38.6 MB | Model_C_JATAYU.pt |
| **Model D** | YOLO11m (Baseline) | Triple-Domain (All 3) | 38.6 MB | Model_D_TripleDomain.pt |
| **YOLO11m-P2** | YOLO11m + P2 Head (Stride 4) | VisDrone | 118.2 MB | yolo11m_p2_visdrone.pt |
| **YOLO11m-P2-NWD**| YOLO11m + P2 + NWD Loss | VisDrone | 118.2 MB | yolo11m_p2_nwd_visdrone.pt |

> *To use pretrained weights, place downloaded .pt files into this weights/ directory.*
