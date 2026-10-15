import os
import logging
import time
import json
from pathlib import Path
import numpy as np
import cv2

# MobileNet là mô hình on-device được thiết kế chạy trên CPU, vô hiệu hóa GPU để đánh giá chuẩn CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Ép buộc TensorFlow chỉ nhận CPU
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


class MobileNetClassifier:
    """Bộ phân loại MobileNetV3 chạy trên CPU (thiết kế on-device)."""

    def __init__(self, model_path, batch_size=10, adult_threshold=0.85):
        """
        - model_path: Đường dẫn tới file .keras
        - batch_size: Số lượng frame xử lý trong một batch.
        - adult_threshold: Ngưỡng xác suất phân loại Adult.
        """
        self.model_path = str(Path(model_path).resolve())
        self.batch_size = batch_size
        self.adult_threshold = adult_threshold
        self.target_size = 224
        self.class_names = ["adult", "normal"]

        logging.info(f"Khởi tạo MobileNetV3 CPU (Direct Resize {self.target_size}x{self.target_size} | Batch frames: {self.batch_size})")
        self._load_model()

    def _load_model(self):
        logging.info(f"Đang tải MobileNet model từ: {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        logging.info("Tải mô hình MobileNetV3 thành công trên CPU.")

    def _preprocess(self, frame_bgr):
        """
        Resize trực tiếp về (224, 224), chuyển sang RGB, NumPy float32 và chuẩn hóa [-1, 1].
        """
        resized = cv2.resize(frame_bgr, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_array = (frame_rgb.astype(np.float32) / 127.5) - 1.0
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

            batch_input = np.array(batch_arrays, dtype=np.float32)
            probabilities = self.model.predict(batch_input, verbose=0)

            for i, sample in enumerate(batch_samples):
                probs = probabilities[i]
                adult_prob = float(probs[0])
                normal_prob = float(probs[1]) if len(probs) > 1 else float(1.0 - adult_prob)
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

        for sample in ssim_stream:
            image_array = self._preprocess(sample["frame_data"])
            batch_samples.append(sample)
            batch_arrays.append(image_array)

            if len(batch_samples) == self.batch_size:
                process_batch()
                batch_samples.clear()
                batch_arrays.clear()
                if stop_early:
                    break

        if not stop_early and batch_samples:
            process_batch()

        elapsed = time.time() - start_time
        adult_count = sum(1 for r in results if r["prediction"] == "adult")
        normal_count = len(results) - adult_count

        logging.info("KẾT QUẢ INFERENCE (MOBILENET CPU):")
        logging.info(f"Tổng frames: {len(results)} | Adult: {adult_count} | Normal: {normal_count} | Thời gian: {elapsed:.2f}s")

        if save_to_disk and output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump({"total_frames": len(results), "results": results}, f, indent=2)

        return results