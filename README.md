# Sensitive-Video-Moderation (PBL6)

Dự án cung cấp hai giải pháp Deep Learning phát hiện và kiểm duyệt video có nội dung người lớn/nhạy cảm:
1. **Pipeline đề xuất (Proposed Method - thư mục `proposed/`)**: Kiến trúc phễu phân tầng 4 giai đoạn (4-Stage Hierarchical Cascade) kết hợp khai thác miền nén I-frame & biến thiên màu sắc HSV, quét thô độ nhạy cao MobileNetV3-Small, phóng đại không gian 360x640 với chọn mẫu đa dạng cực đại (Max-Min Diversity / FPS), và bộ đánh giá hợp nhất hai nhánh (Sustained NSFW & Flash Insert).
2. **Pipeline tham chiếu (Baseline Paper - thư mục `baseline/`)**: Phương pháp từ bài báo tham chiếu dựa trên xử lý dòng (Stream) On-device, kết hợp bộ lọc trùng lặp khung hình SSIM và phân loại thời gian thực bằng MobileNetV3.

---

## Mục Lục
- [Sensitive-Video-Moderation (PBL6)](#sensitive-video-moderation-pbl6)
  - [Mục Lục](#mục-lục)
  - [1. Yêu Cầu Môi Trường \& Cài Đặt](#1-yêu-cầu-môi-trường--cài-đặt)
    - [1.1. Yêu cầu hệ thống](#11-yêu-cầu-hệ-thống)
    - [1.2. Khởi tạo môi trường ảo và cài đặt thư viện](#12-khởi-tạo-môi-trường-ảo-và-cài-đặt-thư-viện)
  - [2. Cấu Hình Biến Môi Trường (.env)](#2-cấu-hình-biến-môi-trường-env)
  - [3. Tóm Tắt Luồng Hoạt Động](#3-tóm-tắt-luồng-hoạt-động)
    - [3.1. Pipeline Đề Xuất (Proposed Method - 4 Giai Đoạn)](#31-pipeline-đề-xuất-proposed-method---4-giai-đoạn)
    - [3.2. Pipeline Tham Chiếu (Baseline Paper)](#32-pipeline-tham-chiếu-baseline-paper)
  - [4. Hướng Dẫn Thực Thi \& Tham Số Dòng Lệnh](#4-hướng-dẫn-thực-thi--tham-số-dòng-lệnh)
    - [4.1. Chạy Pipeline Đề Xuất (Proposed)](#41-chạy-pipeline-đề-xuất-proposed)
      - [Bảng tham số `proposed_adult_video_detect.py`:](#bảng-tham-số-proposed_adult_video_detectpy)
    - [4.2. Cắt Clip Vi Phạm (Proposed)](#42-cắt-clip-vi-phạm-proposed)
    - [4.3. Chạy Pipeline Tham Chiếu (Baseline)](#43-chạy-pipeline-tham-chiếu-baseline)
    - [4.4. Cắt Clip Vi Phạm (Baseline)](#44-cắt-clip-vi-phạm-baseline)

---

## 1. Yêu Cầu Môi Trường & Cài Đặt

### 1.1. Yêu cầu hệ thống
- **Hệ điều hành**: Linux (Ubuntu, Debian, Fedora, Arch) hoặc Windows.
- **Python**: Phiên bản `3.10` trở lên (Khuyên dùng Python 3.11 hoặc 3.12).
- **Phần cứng**: Hỗ trợ chạy trên cả CPU và GPU (NVIDIA GPU với CUDA hỗ trợ tăng tốc tối đa).
- **Yêu cầu bổ sung**: Cài đặt sẵn `ffmpeg` và `ffprobe` trên hệ thống (để phục vụ trích xuất I-frame và cắt video).

### 1.2. Khởi tạo môi trường ảo và cài đặt thư viện
```bash
# 1. Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate      # Trên Linux/macOS
# .venv\Scripts\activate       # Trên Windows

# 2. Cài đặt các thư viện trong requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> Nếu máy tính có GPU NVIDIA, hãy cài đặt PyTorch phiên bản hỗ trợ CUDA để đạt hiệu năng tối ưu:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

---

## 2. Cấu Hình Biến Môi Trường (.env)

Tệp `.env` đặt tại thư mục `src/` quản lý các đường dẫn mô hình và thư mục đầu ra:

```env
# ============================================================
# CẤU HÌNH PIPELINE PROPOSED (PHƯƠNG PHÁP ĐỀ XUẤT)
# ============================================================
RESNET_MODEL=models/resnet50_adult_normal_fcn_layer4_classifier.pth
MOBILENET_MODEL=models/mobilenetv3_small_adult_normal.keras
PROPOSED_MODEL=models/resnet50_adult_normal_fcn_layer4_classifier.pth

PROPOSED_OUT_DIR=../ProposedLog
PROPOSED_CLIPS_OUT_DIR=../proposed_bad_clips
RESNET_OUT_DIR=../ProposedLog
RESNET_CLIPS_OUT_DIR=../proposed_bad_clips

# ============================================================
# CẤU HÌNH PIPELINE BASELINE (PHƯƠNG PHÁP PAPER THAM CHIẾU)
# ============================================================
BASELINE_MODEL=models/mobilenetv3_small_adult_normal.keras

BASELINE_OUT_DIR=../BaselineLog
BASELINE_CLIPS_OUT_DIR=../baseline_bad_clips
MOBILENET_OUT_DIR=../BaselineLog
MOBILENET_CLIPS_OUT_DIR=../baseline_bad_clips
```

---

## 3. Tóm Tắt Luồng Hoạt Động

### 3.1. Pipeline Đề Xuất (Proposed Method - 4 Giai Đoạn)

```
[Video Input]
      │
      ▼
[Giai Đoạn 1: Miền nén & Biến thiên màu sắc]
      ├─ Trích xuất I-frames bằng ffprobe cực nhanh (0-cost CNN)
      ├─ Tính khoảng cách Bhattacharyya trên biểu đồ màu HSV (64x64, 16x16 bins)
      ├─ Tính Z-score đột biến trên sliding window W = 15: Z_flash(t)
      └─ Chọn mẫu: Phi(f_t) = I_iframe(f_t) OR I(delta_t >= 0.5s) OR I(Z_flash(t) > 3.0)
      │
      ▼
[Giai Đoạn 2: Quét thô Độ nhạy cao với MobileNetV3-Small]
      ├─ Resize trực tiếp 224x224, chạy theo lô Batch_Size = 32
      ├─ Nếu p_mobile(t) < 0.35: Loại bỏ sớm (Early Rejection)
      └─ Nếu p_mobile(t) >= 0.35: Đưa vào tập tâm điểm nghi vấn (Anchor Set A)
      │
      ▼
[Kiểm tra Anchor Set A]
      ├─ Nếu A rỗng ──► KẾT THÚC SỚM (Toàn bộ video = NORMAL)
      │
      └─ Nếu có Anchor: Duyệt từng mốc t_0 thuộc A:
            ├─ [Giai Đoạn 3: Phóng đại Không gian & Max-Min Diversity]
            │     ├─ Cửa sổ lân cận: Omega(t_0) = [max(t_0 - 1.0s, start_shot), min(t_0 + 1.0s, end_shot)]
            │     ├─ Chuẩn hóa ảnh xám 32x32 thành vector đơn vị ||v||_2 = 1
            │     ├─ Thuật toán Greedy Farthest Point Sampling chọn tập D* gồm 5 frames
            │     └─ Dự đoán xác suất ResNet-50 FCN ở độ phân giải cao 360x640 (Feature grid 12x20=240)
            │
            └─ [Giai Đoạn 4: Bộ Đánh giá Hợp nhất Hai Nhánh]
                  ├─ Nhánh 1 (Sustained NSFW): P_mean >= 0.85 VÀ tối thiểu 3/5 frames >= 0.80
                  ├─ Nhánh 2 (Flash Insert):
                  │     (1) Đột biến cô lập: Z_flash(t_0) > 3.0 VÀ P_neighbors_max(t_0) < 0.20
                  │     (2) Đồng thuận chéo: p_mobile(t_0) >= 0.90 VÀ p_resnet(f_t0) >= 0.95
                  │     (3) Bất biến phóng đại: p_crop(t_0) = ResNet50_FCN(CenterCrop_360x640(f_t0)) >= 0.95
                  └─ Decision(t_0) = ADULT nếu Nhánh 1 = True HOẶC Nhánh 2 = True
      │
      ▼
[Phán quyết Video: ADULT / NORMAL] ───► Lưu báo cáo 08_final_decision.json
```

### 3.2. Pipeline Tham Chiếu (Baseline Paper)

Chiến lược **Xử lý dòng On-device + Lọc SSIM + Suy luận thời gian thực**:

```
[Video Input]
      │
      ▼
[Bước 1: Frame Extractor Stream] ─────► Đọc tuần tự từng frame dưới dạng Generator trên RAM
      │
      ▼
[Bước 2: SSIM Filter Stream] ─────────► Thu nhỏ ảnh về 224x224 & tính SSIM giữa Frame(t) và Frame(t-1)
      │                                 └─ Loại bỏ khung hình trùng lặp
      ▼
[Bước 3: MobileNetV3 Batch Inference] ─► Gom batch, resize trực tiếp 224x224
      │                                 └─ Dự đoán nhãn Adult / Normal
      ▼
[Bước 4: Evaluator] ──────────────────► Xác định Video = Adult nếu có frame vi phạm
                                        Lưu kết quả vào 12_final_decision.json
```

---

## 4. Hướng Dẫn Thực Thi & Tham Số Dòng Lệnh

### 4.1. Chạy Pipeline Đề Xuất (Proposed)
```bash
python proposed/proposed_adult_video_detect.py --video "../TestDataset/Adult/A_01.mp4"
```
*Để kích hoạt dừng sớm ngay khi phát hiện vi phạm Adult đầu tiên:*
```bash
python proposed/proposed_adult_video_detect.py --video "../TestDataset/Adult/A_01.mp4" --early_stop
```

#### Bảng tham số `proposed_adult_video_detect.py`:
| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `--video` | `str` | *Bắt buộc* | Đường dẫn đến file video cần phân tích |
| `--early_stop` | `flag` | `False` | Dừng sớm ngay khi phát hiện anchor vi phạm Adult đầu tiên |
| `--out_dir` | `str` | Lấy từ `.env` | Thư mục lưu kết quả JSON và log (`../ProposedLog`) |
| `--mobilenet_model` | `str` | Lấy từ `.env` | Đường dẫn mô hình MobileNetV3-Small (`.keras`) |
| `--resnet_model` | `str` | Lấy từ `.env` | Đường dẫn mô hình ResNet50 FCN (`.pth`) |
| `--delta_t` | `float` | `0.5` | Khoảng thời gian tối đa giữa các frame lấy mẫu cơ sở |
| `--z_threshold` | `float` | `3.0` | Ngưỡng Z-score phát hiện đột biến flash nhấp nháy |
| `--rejection_threshold` | `float` | `0.35` | Ngưỡng loại bỏ sớm của MobileNetV3 ở Bước 2 |
| `--batch_size` | `int` | `32` | Số lượng frame gom trong một batch suy luận MobileNet |
| `--save_intermediate` | `flag` | `False` | Lưu log đĩa và các frame ảnh ở từng giai đoạn để debug |

---

### 4.2. Cắt Clip Vi Phạm (Proposed)
Sau khi có file `08_final_decision.json`, trích xuất các đoạn video nhạy cảm bằng:
```bash
python proposed/cut_adult_regions.py --json "../ProposedLog/A_01/08_final_decision.json"
```

---

### 4.3. Chạy Pipeline Tham Chiếu (Baseline)
```bash
python baseline/mobilenet_adult_video_detect.py --video "../TestDataset/Normal/N_01.mp4"
```
*Để kích hoạt dừng sớm ngay khi gặp frame nhạy cảm đầu tiên:*
```bash
python baseline/mobilenet_adult_video_detect.py --video "../TestDataset/Normal/N_01.mp4" --early_stop
```

---

### 4.4. Cắt Clip Vi Phạm (Baseline)
Sau khi có file `12_final_decision.json`:
```bash
python baseline/cut_mobilenet_regions.py --json "../BaselineLog/N_01/12_final_decision.json"
```
