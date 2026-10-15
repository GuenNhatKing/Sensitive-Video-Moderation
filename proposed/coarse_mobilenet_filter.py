import os
import json
import logging
from pathlib import Path
import numpy as np
import cv2

# MobileNetV3-Small chạy quét thô trên CPU (thiết kế on-device)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


class CoarseMobileNetFilter:
    """
    Giai đoạn 2: Quét thô Độ nhạy cao bằng MobileNetV3-Small trên CPU
    - Mục tiêu: Lọc bỏ nhanh 85% - 90% khung hình bình thường bằng mô hình siêu nhẹ (0.94M params),
      giữ lại 100% các tâm điểm nghi vấn (Recall = 100%).
    - Hàm lọc phễu: p_mobile(t) = MobileNetV3_Small( resize(f_t, 224, 224) )
      IF p_mobile(t) < 0.35:
          NORMAL (Loại bỏ sớm - Early Rejection)
      ELSE (p_mobile(t) >= 0.35):
          Anchor_Set A = A U {t_0} -> Chuyển sang Bước 3
    """

    def __init__(self, model_path, batch_size=32, rejection_threshold=0.35):
        self.model_path = str(Path(model_path).resolve())
        self.batch_size = batch_size
        self.rejection_threshold = rejection_threshold
        self.target_size = (224, 224)

        logging.info(
            f"Khởi tạo CoarseMobileNetFilter CPU (Batch={self.batch_size}, "
            f"Threshold={self.rejection_threshold}, Input={self.target_size})"
        )
        self._load_model()

    def _load_model(self):
        logging.info(f"Đang tải MobileNetV3-Small (CPU) từ: {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình MobileNet: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        logging.info("Tải MobileNetV3-Small thành công trên CPU.")

    def warmup(self):
        """
        Chạy nháp (Warm-up pass) với dummy tensor để nạp graph và khởi tạo kernel trước khi đo.
        """
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        logging.info("MobileNetV3-Small Warm-up pass hoàn tất trên RAM.")

    def _preprocess_frame(self, frame_bgr):
        resized = cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_array = (frame_rgb.astype(np.float32) / 127.5) - 1.0
        return image_array

    def filter_candidates(self, samples, output_json_path=None, save_to_disk=False):
        """
        Dự đoán hàng loạt các candidate frames từ Stage 1.
        Phân loại Early Rejection (p_mobile < 0.35) và gom Anchor Set A (p_mobile >= 0.35).
        """
        if not samples:
            logging.info("Stage 2: Không có candidate frame nào từ Stage 1.")
            return [], []

        logging.info(f"Stage 2: Bắt đầu quét thô {len(samples)} frames với MobileNetV3-Small (CPU)...")

        batch_arrays = [self._preprocess_frame(s["frame_data"]) for s in samples]
        batch_input = np.array(batch_arrays, dtype=np.float32)

        probabilities = self.model.predict(batch_input, batch_size=self.batch_size, verbose=0)

        all_results = []
        anchor_set = []

        for idx, sample in enumerate(samples):
            probs = probabilities[idx]
            p_mobile = float(probs[0])
            p_normal = float(probs[1]) if len(probs) > 1 else float(1.0 - p_mobile)

            is_early_rejected = bool(p_mobile < self.rejection_threshold)
            is_anchor = not is_early_rejected

            result_entry = {
                "sample_id": sample["sample_id"],
                "frame_index": sample["frame_index"],
                "time_sec": sample["time_sec"],
                "is_iframe": sample["is_iframe"],
                "is_flash_candidate": sample["is_flash_candidate"],
                "z_flash": sample["z_flash"],
                "d_bhattacharyya": sample["d_bhattacharyya"],
                "shot_id": sample["shot_id"],
                "start_shot": sample["start_shot"],
                "end_shot": sample["end_shot"],
                "p_mobile": round(p_mobile, 6),
                "p_normal": round(p_normal, 6),
                "status": "EARLY_REJECTED" if is_early_rejected else "ANCHOR_CANDIDATE",
                "image_path": sample.get("image_path")
            }
            all_results.append(result_entry)

            if is_anchor:
                anchor_entry = dict(sample)
                anchor_entry["p_mobile"] = round(p_mobile, 6)
                anchor_entry["p_normal"] = round(p_normal, 6)
                anchor_set.append(anchor_entry)

        rejection_rate = (len(samples) - len(anchor_set)) / len(samples) * 100 if samples else 0.0
        logging.info(
            f"Stage 2 hoàn thành: Loại bỏ sớm {len(samples) - len(anchor_set)}/{len(samples)} frames ({rejection_rate:.1f}%). "
            f"Số tâm điểm nghi vấn (Anchors): {len(anchor_set)}."
        )

        if save_to_disk and output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump({
                    "total_candidates": len(samples),
                    "rejected_count": len(samples) - len(anchor_set),
                    "anchor_count": len(anchor_set),
                    "rejection_threshold": self.rejection_threshold,
                    "results": all_results
                }, f, indent=2, ensure_ascii=False)
            logging.info(f"Đã lưu kết quả Stage 2: {output_json_path}")

        return all_results, anchor_set
