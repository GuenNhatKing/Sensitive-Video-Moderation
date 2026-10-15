import os
import subprocess
import logging
import json
import numpy as np
import cv2
from pathlib import Path


class CompressedColorSampler:
    """
    THAY ĐỔI 1: Khai thác Miền nén & Biến thiên Màu sắc
    - Trích xuất I-frame trực tiếp từ packet container (nhanh, 0-cost CNN).
    - Đo khoảng cách Bhattacharyya trên biểu đồ màu HSV (64x64, 16x16 bins).
    - Tính Z-score trên sliding window W = 15 frames để phát hiện đột biến flash nhấp nháy.
    - Quyết định mẫu: Phi(f_t) = I_iframe(f_t) OR I(delta_t >= 0.5s) OR I(Z_flash(t) > 3.0).
    """

    def __init__(self, delta_t=0.5, z_threshold=3.0, window_size=15):
        self.delta_t = delta_t
        self.z_threshold = z_threshold
        self.window_size = window_size
        logging.info(
            f"Khởi tạo CompressedColorSampler (delta_t={self.delta_t}s, "
            f"Z_threshold={self.z_threshold}, Window={self.window_size})"
        )

    def extract_iframe_timestamps(self, video_path):
        """
        Trích xuất timestamp các I-frames bằng ffprobe cực nhanh (chỉ đọc packet header).
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v",
            "-show_packets",
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0",
            str(video_path)
        ]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8")
        except Exception as e:
            logging.warning(f"Lỗi khi chạy ffprobe trích xuất I-frame: {e}. Tiếp tục bằng lấy mẫu thông thường.")
            return []

        iframe_timestamps = []
        for line in output.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and "K" in parts[1]:
                try:
                    ts = float(parts[0])
                    iframe_timestamps.append(ts)
                except ValueError:
                    pass

        return sorted(list(set(iframe_timestamps)))

    def build_shots_from_iframes(self, iframe_timestamps, duration_sec):
        """
        Phân chia shot video dựa trên ranh giới I-frames hoặc [0, duration_sec].
        """
        # Đảm bảo có mốc 0.0 và duration_sec
        points = [0.0]
        for ts in iframe_timestamps:
            if 0.0 < ts < duration_sec:
                points.append(ts)
        points.append(duration_sec)
        points = sorted(list(set(points)))

        shots = []
        for i in range(len(points) - 1):
            s_start = round(points[i], 3)
            s_end = round(points[i + 1], 3)
            shots.append({
                "shot_id": i,
                "start_sec": s_start,
                "end_sec": s_end,
                "duration_sec": round(s_end - s_start, 3)
            })

        if not shots:
            shots = [{
                "shot_id": 0,
                "start_sec": 0.0,
                "end_sec": round(duration_sec, 3),
                "duration_sec": round(duration_sec, 3)
            }]
        return shots

    def find_shot_for_time(self, time_sec, shots):
        """
        Tìm thông tin shot tương ứng với thời điểm time_sec.
        """
        for shot in shots:
            if shot["start_sec"] <= time_sec <= shot["end_sec"]:
                return shot
        return shots[-1]

    def sample_video(self, video_path, output_dir=None, save_to_disk=False):
        """
        Quét toàn bộ video theo luồng tuần tự, tính biến thiên màu sắc HSV và Z-score flash.
        Trả về danh sách các frame thỏa mãn hàm quyết định Phi(f_t) = 1.
        """
        video_path = Path(video_path).resolve()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        logging.info(f"Phân tích video: {video_path.name} ({total_frames} frames, {fps:.2f} FPS, {duration_sec:.2f}s)")

        # 1. Trích xuất I-frame timestamps
        iframe_timestamps = self.extract_iframe_timestamps(video_path)
        shots = self.build_shots_from_iframes(iframe_timestamps, duration_sec)
        iframe_indices = set([int(round(ts * fps)) for ts in iframe_timestamps])
        logging.info(f"Đã phát hiện {len(iframe_indices)} I-frames -> Phân chia thành {len(shots)} shots.")

        if save_to_disk and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        samples = []
        flash_candidates_count = 0
        prev_hist = None
        db_window = []
        t_last = -999999.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time_sec = frame_idx / fps

            # Thu nhỏ ảnh về 64x64 và tính biểu đồ màu HSV 16x16 bins
            small_bgr = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            small_hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([small_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist = hist / (hist.sum() + 1e-7)

            # Tính khoảng cách Bhattacharyya giữa frame t và frame t-1
            if prev_hist is None:
                d_B = 0.0
            else:
                bc = np.sum(np.sqrt(hist * prev_hist))
                d_B = float(np.sqrt(max(0.0, 1.0 - bc)))
            prev_hist = hist

            # Tính Z-score flash trên cửa sổ trượt W = 15 frames
            if len(db_window) >= 2:
                mean_w = float(np.mean(db_window))
                std_w = float(np.std(db_window))
                z_flash = (d_B - mean_w) / (std_w + 1e-6)
            else:
                z_flash = 0.0

            db_window.append(d_B)
            if len(db_window) > self.window_size:
                db_window.pop(0)

            is_flash_candidate = bool(z_flash > self.z_threshold)
            is_iframe = bool((frame_idx in iframe_indices) or (frame_idx == 0))
            delta_t = (frame_idx - t_last) / fps

            # Hàm quyết định: Phi(f_t) = I_iframe(f_t) OR I(delta_t >= 0.5s) OR I(Z_flash(t) > 3.0)
            phi = is_iframe or (delta_t >= self.delta_t) or is_flash_candidate

            if phi:
                t_last = frame_idx
                if is_flash_candidate:
                    flash_candidates_count += 1

                shot = self.find_shot_for_time(time_sec, shots)

                img_path = None
                if save_to_disk and output_dir:
                    filename = f"frame_{frame_idx:06d}_{time_sec:.2f}s.jpg"
                    img_path = os.path.join(output_dir, filename)
                    cv2.imwrite(img_path, frame)

                sample_meta = {
                    "sample_id": len(samples) + 1,
                    "frame_index": frame_idx,
                    "time_sec": round(time_sec, 4),
                    "is_iframe": is_iframe,
                    "is_flash_candidate": is_flash_candidate,
                    "z_flash": round(float(z_flash), 4),
                    "d_bhattacharyya": round(float(d_B), 4),
                    "shot_id": shot["shot_id"],
                    "start_shot": shot["start_sec"],
                    "end_shot": shot["end_sec"],
                    "image_path": img_path,
                    "frame_data": frame  # Giữ trên RAM cho các bước tiếp theo
                }
                samples.append(sample_meta)

            frame_idx += 1

        cap.release()
        logging.info(
            f"Stage 1 hoàn thành: Lấy {len(samples)}/{total_frames} frames "
            f"({len(samples)/total_frames*100:.1f}%), trong đó có {flash_candidates_count} flash candidates."
        )

        video_info = {
            "video_path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "selected_frames": len(samples),
            "flash_candidates": flash_candidates_count,
            "shots": shots
        }
        return samples, shots, video_info
