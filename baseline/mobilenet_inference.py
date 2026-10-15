import os
import site
import glob
import ctypes

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Tự động nạp các thư viện liên kết động CUDA/cuDNN trong virtual environment (nếu có)
try:
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_dir):
            for so_file in sorted(glob.glob(os.path.join(nvidia_dir, "*/lib/*.so*"))):
                try:
                    ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
except Exception:
    pass

import logging
import time
import json
import numpy as np
import cv2

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Bật Dynamic Memory Growth cho GPU để tránh chiếm trọn VRAM một lúc
try:
    _gpus = tf.config.list_physical_devices("GPU")
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass


class MobileNetClassifier:
    def __init__(self, model_path, batch_size=10, adult_threshold=0.85):
        """
        - model_path: Đường dẫn tới file .keras
        - batch_size: Số lượng frame xử lý trong một batch.
        - adult_threshold: Ngưỡng xác suất phân loại Adult.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.adult_threshold = adult_threshold
        self.target_size = 224
        self.class_names = ["adult", "normal"]
        
        logging.info(f"Khởi tạo MobileNetV3 (Direct Resize {self.target_size}x{self.target_size} | Batch frames: {self.batch_size})")
        
        self._load_model()
        
    def _load_model(self):
        logging.info(f"Đang tải MobileNet model từ {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình: {self.model_path}")
            
        self.model = tf.keras.models.load_model(self.model_path)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                dev_label = details.get("device_name", "GPU")
                device_name = f"GPU ({dev_label})"
            except Exception:
                device_name = "GPU"
        else:
            device_name = "CPU"
        logging.info(f"Tải mô hình thành công. Thiết bị thực thi: {device_name}")

    def _preprocess(self, frame_bgr):
        """
        Resize trực tiếp về (224, 224) (chấp nhận méo ảnh),
        chuyển sang RGB, NumPy float32 và chuẩn hóa [-1, 1].
        Trả về mảng NumPy shape (224, 224, 3).
        """
        resized = cv2.resize(frame_bgr, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_array = frame_rgb.astype(np.float32)
        image_array = (image_array / 127.5) - 1.0
        return image_array

    def predict_stream(self, ssim_stream, output_json_path=None, save_to_disk=False, early_stop=False):
        results = []
        batch_samples = []
        batch_arrays = []
        stop_early = False
        
        start_time = time.time()
        
        def process_batch():
            nonlocal stop_early
            if not batch_arrays:
                return
                
            # Gộp batch: Shape sẽ là (N, 224, 224, 3)
            batch_input = np.array(batch_arrays)
            
            # Inference qua mô hình MobileNet
            probabilities = self.model.predict(batch_input, verbose=0)
            
            # Xử lý kết quả cho từng frame
            for i, sample in enumerate(batch_samples):
                probs = probabilities[i]
                
                adult_prob = float(probs[0])
                normal_prob = float(probs[1])
                
                # Phân loại dựa trên ngưỡng xác suất cấu hình nhằm hạn chế False Positive
                prediction = "adult" if adult_prob >= self.adult_threshold else "normal"
                
                results.append({
                    "frame_index": sample["frame_index"],
                    "time_sec": sample["time_sec"],
                    "prediction": prediction,
                    "adult_probability": round(adult_prob, 6),
                    "normal_probability": round(normal_prob, 6)
                })
                
                if early_stop and prediction == "adult":
                    stop_early = True
                    logging.info(
                        f"Phát hiện frame {sample['frame_index']} ({sample['time_sec']:.2f}s) là ADULT "
                        f"(xác suất {adult_prob:.4f}). [Early Stop kích hoạt: Dừng sớm pipeline]"
                    )
                    break

        # Tiêu thụ luồng dữ liệu
        for sample in ssim_stream:
            # Tiền xử lý frame: resize trực tiếp về 224x224 và chuẩn hóa
            image_array = self._preprocess(sample["frame_data"])
            
            batch_samples.append(sample)
            batch_arrays.append(image_array)
            
            # Kiểm tra theo số lượng FRAME
            if len(batch_samples) == self.batch_size:
                process_batch()
                batch_samples.clear()
                batch_arrays.clear()
                if stop_early:
                    break
                
        # Xử lý nốt những frame lẻ còn sót lại nếu chưa dừng sớm
        if not stop_early and batch_samples:
            process_batch()
        
        elapsed = time.time() - start_time
        
        # Tổng kết
        adult_count = sum(1 for r in results if r["prediction"] == "adult")
        normal_count = len(results) - adult_count
        
        logging.info("KẾT QUẢ INFERENCE (MOBILENET):")
        logging.info(f"Tổng frames: {len(results)} | Adult: {adult_count} | Normal: {normal_count} | Thời gian: {elapsed:.2f}s")

        if save_to_disk and output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump({"total_frames": len(results), "results": results}, f, indent=2)
                
        return results