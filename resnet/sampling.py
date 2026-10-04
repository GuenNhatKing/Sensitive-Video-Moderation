import cv2
import os
import math
import json
import logging
import numpy as np

class VideoSampler:
    def __init__(self, delta_t=0.5, n_min=2, n_max=None):
        self.delta_t = delta_t
        self.n_min = n_min
        self.n_max = n_max

    def sample_shots(self, video_path, shots, output_dir=None, save_to_disk=True):
        """
        Lấy mẫu frame từ danh sách các cảnh (shots).
        - Nếu save_to_disk=True: Lưu ảnh ra thư mục output_dir.
        - Trả về danh sách metadata kèm dữ liệu frame.
        """
        if not shots:
            logging.warning("Không có cảnh (shots) nào để lấy mẫu.")
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video để lấy mẫu: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_samples = 0
        sample_metadata = []

        if save_to_disk and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Các frame mẫu sẽ được lưu tại: {output_dir}")

        for shot in shots:
            shot_id = shot["shot_id"]
            start_frame = shot["start_frame"]
            end_frame = shot["end_frame"]
            duration = shot["duration_sec"]

            # Tính toán số lượng mẫu theo khoảng thời gian và giới hạn cấu hình
            n_samples = math.ceil(duration / self.delta_t)
            if self.n_max is not None:
                n_samples = min(self.n_max, n_samples)
            n_samples = max(self.n_min, n_samples)
            
            available_frames = end_frame - start_frame + 1
            n_samples = min(n_samples, available_frames)

            # Tạo danh sách vị trí frame, tránh hai đầu biên
            if n_samples > 0:
                frame_positions = np.linspace(start_frame, end_frame, n_samples + 2)[1:-1]
                frame_positions = np.unique(np.round(frame_positions).astype(int))
            else:
                frame_positions = []

            logging.debug(f"Shot {shot_id:02d} ({duration:.3f}s) -> Lấy {len(frame_positions)} frames")

            shot_dir = None
            if save_to_disk and output_dir:
                shot_dir = os.path.join(output_dir, f"shot_{shot_id:03d}")
                os.makedirs(shot_dir, exist_ok=True)

            for sample_id, frame_idx in enumerate(frame_positions, start=1):
                frame_idx = int(frame_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logging.warning(f"Không thể đọc frame {frame_idx} tại shot {shot_id}")
                    continue

                time_sec = frame_idx / fps
                output_path = None

                # Lưu ra đĩa cứng nếu có yêu cầu
                if save_to_disk and shot_dir:
                    filename = f"sample_{sample_id:02d}_frame_{frame_idx:05d}_time_{time_sec:.2f}s.jpg"
                    output_path = os.path.join(shot_dir, filename)
                    cv2.imwrite(output_path, frame)

                # Lưu metadata (Có thể đính kèm dữ liệu frame RGB để truyền thẳng lên RAM)
                sample_metadata.append({
                    "shot_id": shot_id,
                    "sample_id": sample_id,
                    "frame_index": frame_idx,
                    "time_sec": round(time_sec, 4),
                    "image_path": output_path, # Sẽ là None nếu không lưu đĩa
                    "frame_data": frame if not save_to_disk else None # Lưu ảnh trên RAM để bước sau dùng
                })
                total_samples += 1

        cap.release()
        logging.info(f"Hoàn thành lấy mẫu. Tổng số frames thu thập: {total_samples}")

        # Lưu log json nếu cần
        if save_to_disk and output_dir:
            metadata = {
                "video": str(video_path),
                "fps": fps,
                "sampling": {"delta_t": self.delta_t, "n_min": self.n_min, "n_max": self.n_max},
                "shot_count": len(shots),
                "total_samples": total_samples,
                "samples": [{k: v for k, v in m.items() if k != 'frame_data'} for m in sample_metadata]
            }
            metadata_path = os.path.join(output_dir, "samples.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        return sample_metadata