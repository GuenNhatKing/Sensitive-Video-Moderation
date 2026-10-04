# PBL6: Hệ Thống Kiểm Duyệt Nội Dung Video Nhạy Cảm (Adult Video Detection)

Dự án cung cấp hai giải pháp Deep Learning phát hiện và kiểm duyệt video có nội dung người lớn/nhạy cảm:
1. **Pipeline ResNet-50 kết hợp TransNetV2**: Phân tích đa tầng theo phân đoạn cảnh (Shot Boundary Detection), kết hợp lấy mẫu thưa (Sparse) - dày (Dense) và kiểm tra tính nhất quán thời gian (Temporal Consistency).
2. **Pipeline MobileNetV3**: Xử lý dòng (Stream) tối ưu cho môi trường thiết bị hạn chế tài nguyên (On-device), kết hợp bộ lọc trùng lặp khung hình SSIM (Structural Similarity Index).

---

## Mục Lục
- [1. Yêu Cầu Môi Trường & Cài Đặt](#1-yêu-cầu-môi-trường--cài-đặt)
- [2. Hướng Dẫn TransNetV2 & Chuyển Đổi Trọng Số (Weights)](#2-hướng-dẫn-transnetv2--chuyển-đổi-trọng-số-weights)
- [3. Cấu Hình Biến Môi Trường (.env)](#3-cấu-hình-biến-môi-trường-env)
- [4. Tóm Tắt Luồng Hoạt Động](#4-tóm-tắt-luồng-hoạt-động)
  - [4.1. Luồng Pipeline ResNet-50 + TransNetV2](#41-luồng-pipeline-resnet-50--transnetv2)
  - [4.2. Luồng Pipeline MobileNetV3](#42-luồng-pipelin-mobilenetv3)
- [5. Hướng Dẫn Thực Thi & Tham Số Dòng Lệnh](#5-hướng-dẫn-thực-thi--tham-số-dòng-lệnh)
  - [5.1. Chạy Pipeline ResNet-50](#51-chạy-pipeline-resnet-50)
  - [5.2. Cắt Clip Vi Phạm (ResNet)](#52-cắt-clip-vi-phạm-resnet)
  - [5.3. Chạy Pipeline MobileNetV3](#53-chạy-pipeline-mobilenetv3)
  - [5.4. Cắt Clip Vi Phạm (MobileNet)](#54-cắt-clip-vi-phạm-mobilenet)

---

## 1. Yêu Cầu Môi Trường & Cài Đặt

### 1.1. Yêu cầu hệ thống
- **Hệ điều hành**: Linux (Ubuntu/Debian, Fedora, Arch) hoặc Windows.
- **Python**: Phiên bản `3.10` trở lên (Khuyên dùng Python 3.11 hoặc 3.12).
- **Phần cứng**: Hỗ trợ chạy trên cả CPU và GPU (NVIDIA GPU với CUDA hỗ trợ tăng tốc tối đa).
- **Yêu cầu bổ sung**: Cài đặt sẵn `ffmpeg` trên hệ thống (để phục vụ cắt video).

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

## 2. Hướng Dẫn TransNetV2 & Chuyển Đổi Trọng Số (Weights)

TransNetV2 là mô hình nhận diện ranh giới phân cảnh (Shot Boundary Detection) được phát triển gốc trên TensorFlow/Keras. Pipeline ResNet trong dự án sử dụng phiên bản PyTorch để chạy đồng bộ trên GPU CUDA.

### Cách 1: Sử dụng thư viện pip (Khuyên dùng)
Dự án đã cấu hình `transnetv2-pytorch` trong `requirements.txt`. Khi cài xong `requirements.txt`, mã nguồn sẽ tự động nạp module này mà không cần clone repo TransNetV2 thủ công.

### Cách 2: Tải TransNetV2 từ GitHub và tự Convert Weights sang PyTorch
Nếu bạn muốn tự tải mã nguồn gốc và chuyển đổi trọng số từ checkpoint TensorFlow sang `.pth`:

1. **Clone repository TransNetV2:**
   ```bash
   git clone https://github.com/soCzech/TransNetV2.git
   ```

2. **Chuyển đổi trọng số TensorFlow sang PyTorch:**
   Thư mục `TransNetV2/inference-pytorch` đã cung cấp sẵn script chuyển đổi `convert_weights.py`. Chạy lệnh sau:
   ```bash
   cd TransNetV2/inference-pytorch
   python convert_weights.py --tf_weights ../inference/transnetv2-weights/
   ```
   *Script sẽ trích xuất trọng số TensorFlow từ thư mục `transnetv2-weights/`, re-map tên các layer sang cấu trúc PyTorch, kiểm tra tính tương đương và lưu file `transnetv2-pytorch-weights.pth`.*

3. **Chép file trọng số vào thư mục `models/` của dự án:**
   ```bash
   # Từ thư mục gốc dự án (src)
   cp TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth models/
   ```

---

## 3. Cấu Hình Biến Môi Trường (.env)

Tệp `.env` đặt tại thư mục `src/` quản lý các đường dẫn mô hình và thư mục đầu ra. Cấu trúc chuẩn:

```env
# ============================================================
# CẤU HÌNH PIPELINE RESNET
# ============================================================
TRANSNET_WEIGHTS=models/transnetv2-pytorch-weights.pth
TRANSNET_DIR=TransNetV2

RESNET_MODEL=models/resnet50_adult_normal_fcn_layer4_classifier.pth
RESNET_OUT_DIR=../ResnetLog
RESNET_CLIPS_OUT_DIR=../resnet_bad_clips

# ============================================================
# CẤU HÌNH PIPELINE MOBILENET
# ============================================================
MOBILENET_MODEL=models/mobilenetv3_small_adult_normal.keras
MOBILENET_OUT_DIR=../MobilenetLog
MOBILENET_CLIPS_OUT_DIR=../mobilenet_bad_clips
```

---

## 4. Tóm Tắt Luồng Hoạt Động

### 4.1. Luồng Pipeline ResNet-50 + TransNetV2

Chiến lược kết hợp **Phân cảnh + Lấy mẫu thưa/dày + Nhất quán thời gian**:

```
[Video Input]
      │
      ▼
[Bước 1: TransNetV2 Shot Detection] ──► Phân chia video thành các Shot độc lập
      │
      ▼
[Bước 2: Sparse Sampling] ────────────► Trích xuất mẫu thưa (0.5s/frame) theo từng shot
      │
      ▼
[Bước 3: Sparse ResNet-50] ───────────► Dự đoán xác suất Adult trên các frame thưa
      │
      ▼
[Bước 4: Suspicious Region Detection] ► Gom các frame nghi vấn gần nhau thành Vùng (Region)
      │                                 ├─ Nếu frame vượt ngưỡng tin cậy cao: Đánh dấu Direct Adult
      │                                 └─ Các vùng còn lại: Đưa vào danh sách cần Dense Sampling
      ▼
[Kiểm tra & Phân nhánh]
      ├─ Nếu bật --early_stop và có Direct Adult ──► KẾT THÚC SỚM (Bỏ qua Dense)
      │
      └─ Duyệt tuần tự các vùng nghi vấn:
            ├─ [Bước 5: Dense Sampling] ────────► Lấy mẫu dày (0.1s/frame, ±1.0s quanh vùng)
            ├─ [Bước 6: Dense ResNet-50] ───────► Suy luận trên các frame dày mới
            ├─ [Bước 7: Gộp Thưa + Dày] ────────► Đồng bộ khung thời gian thực tế của vùng
            └─ [Bước 8: Final Evaluation] ──────► Đánh giá Dynamic Top-K & Temporal Consistency
                  ├─ Nếu vùng thỏa mãn Adult và bật --early_stop ──► KẾT THÚC SỚM
                  └─ Ngược lại: Xét tiếp vùng kế tiếp cho đến hết
      │
      ▼
[Phán quyết Video: ADULT / NORMAL] ───► Lưu báo cáo 08_final_decision.json
```

* **Dynamic Top-K**: Lấy trung bình các frame có xác suất cao nhất của vùng (tỷ lệ cấu hình theo tổng số frame).
* **Temporal Consistency (Tính nhất quán thời gian)**: Yêu cầu chuỗi vi phạm liên tục trong cùng một shot, với khoảng cách tối đa cho phép được tính động:
  $$\text{max\_gap\_sec} = \text{required\_consistent\_frames} \times \text{sampling\_interval\_sec}$$

---

### 4.2. Luồng Pipeline MobileNetV3

Chiến lược **Xử lý dòng On-device + Lọc SSIM + Suy luận thời gian thực**:

```
[Video Input]
      │
      ▼
[Bước 1: Frame Extractor Stream] ─────► Đọc tuần tự từng frame dưới dạng Generator trên RAM
      │
      ▼
[Bước 2: SSIM Filter Stream] ─────────► Tính Structural Similarity giữa Frame(t) và Frame(t-1)
      │                                 └─ Loại bỏ khung hình trùng lặp (giảm 10% - 30% tải)
      ▼
[Bước 3: MobileNetV3 Batch Inference] ─► Gom batch (16 frames), resize trực tiếp 224x224
      │                                 ├─ Tự động nạp CUDA & kích hoạt GPU (Dynamic Memory Growth)
      │                                 └─ Dự đoán nhãn Adult / Normal
      ▼
[Cơ chế Early Stop]
      ├─ Nếu bật --early_stop và gặp frame Adult đầu tiên:
      │     └─ Ngắt Generator ngay lập tức, dừng giải mã video và báo cáo kết quả vi phạm
      │
      └─ Nếu không bật: Quét hết toàn bộ video
      │
      ▼
[Bước 4: Evaluator] ──────────────────► Xác định Video = Adult nếu có frame vi phạm
                                        Lưu kết quả vào 12_final_decision.json
```

---

## 5. Hướng Dẫn Thực Thi & Tham Số Dòng Lệnh

### 5.1. Chạy Pipeline ResNet-50
```bash
python resnet/resnet_adult_video_detect.py --video "../Test/adult-1.mp4"
```
*Để kích hoạt dừng sớm ngay khi phát hiện vùng vi phạm đầu tiên:*
```bash
python resnet/resnet_adult_video_detect.py --video "../Test/adult-1.mp4" --early_stop
```

#### Bảng tham số `resnet_adult_video_detect.py`:
| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `--video` | `str` | *Bắt buộc* | Đường dẫn đến file video cần phân tích |
| `--early_stop` | `flag` | `False` | Dừng sớm ngay khi phát hiện vùng Adult đầu tiên |
| `--out_dir` | `str` | Lấy từ `.env` | Thư mục lưu kết quả JSON và log |
| `--transnet_weights` | `str` | Lấy từ `.env` | Đường dẫn file trọng số TransNetV2 (`.pth`) |
| `--transnet_dir` | `str` | Lấy từ `.env` | Thư mục mã nguồn TransNetV2 (nếu dùng clone) |
| `--resnet_model` | `str` | Lấy từ `.env` | Đường dẫn mô hình ResNet50 Classifier (`.pth`) |
| `--save_intermediate` | `flag` | `False` | Lưu log đĩa ở từng bước trung gian để debug |

---

### 5.2. Cắt Clip Vi Phạm (ResNet)
Sau khi có file `08_final_decision.json`, trích xuất các đoạn video nhạy cảm bằng:
```bash
python resnet/cut_adult_regions.py --json "../ResnetLog/adult-1/08_final_decision.json"
```

#### Bảng tham số `cut_adult_regions.py`:
| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `--json` | `str` | *Bắt buộc* | Đường dẫn đến file `08_final_decision.json` |
| `--out_dir` | `str` | Lấy từ `.env` | Thư mục lưu các clip video được cắt |
| `--video` | `str` | `None` | Ghi đè đường dẫn video gốc nếu file JSON bị di chuyển |
| `--pad` | `float` | `0.0` | Mở rộng thêm X giây ở hai đầu mỗi đoạn cắt |

---

### 5.3. Chạy Pipeline MobileNetV3
```bash
python mobilenet/mobilenet_adult_video_detect.py --video "../Test/normal-1.mp4"
```
*Để kích hoạt dừng sớm ngay khi gặp frame nhạy cảm đầu tiên:*
```bash
python mobilenet/mobilenet_adult_video_detect.py --video "../Test/normal-1.mp4" --early_stop
```

#### Bảng tham số `mobilenet_adult_video_detect.py`:
| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `--video` | `str` | *Bắt buộc* | Đường dẫn đến file video cần phân tích |
| `--early_stop` | `flag` | `False` | Dừng sớm ngay khi phát hiện frame Adult đầu tiên |
| `--out_dir` | `str` | Lấy từ `.env` | Thư mục lưu kết quả JSON |
| `--model` | `str` | Lấy từ `.env` | Đường dẫn file mô hình MobileNet (`.keras`) |
| `--ssim` | `float` | `0.95` | Ngưỡng tương đồng SSIM để loại bỏ frame trùng |
| `--adult_threshold` | `float` | `0.85` | Ngưỡng xác suất để phân loại một frame là Adult |
| `--batch_size` | `int` | `16` | Số lượng frame gom trong một batch suy luận |
| `--save_intermediate`| `flag` | `False` | Lưu các frame ảnh đã trích xuất/lọc ra đĩa |

---

### 5.4. Cắt Clip Vi Phạm (MobileNet)
Sau khi có file `12_final_decision.json`, gom các frame vi phạm thành vùng và cắt video:
```bash
python mobilenet/cut_mobilenet_regions.py --json "../MobilenetLog/normal-1/12_final_decision.json"
```

#### Bảng tham số `cut_mobilenet_regions.py`:
| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `--json` | `str` | *Bắt buộc* | Đường dẫn đến file `12_final_decision.json` |
| `--out_dir` | `str` | Lấy từ `.env` | Thư mục lưu clip cắt |
| `--video` | `str` | `None` | Ghi đè vị trí video gốc nếu cần |
| `--gap` | `float` | `1.5` | Khoảng cách thời gian tối đa giữa các frame để gom cụm |
| `--pad` | `float` | `1.0` | Mở rộng thêm X giây ở hai đầu vùng cắt |
