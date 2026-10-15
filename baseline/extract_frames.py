import cv2
import time
import logging
import os
from pathlib import Path

class FrameExtractor:
    def __init__(self):
        logging.info("Khởi tạo Frame Extractor (Trích xuất toàn bộ frames)")

    def extract_stream(self, video_path, output_dir=None, save_to_disk=False):
        """
        Trích xuất toàn bộ số frame từ video sử dụng generator stream để tối ưu bộ nhớ.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        if width > 0 and height > 0:
            ratio = width / height
            allowed_ratios = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
            is_valid = any(abs(ratio - target) <= 0.08 for target in allowed_ratios.values())
            if not is_valid:
                cap.release()
                raise ValueError(
                    f"Aspect ratio không hợp lệ: {width}x{height} (ratio={ratio:.4f}). "
                    f"Chỉ chấp nhận 3 loại tỉ lệ: 16:9, 9:16, hoặc 1:1."
                )

        logging.info(f"Thông tin Video: {width}x{height} | {video_fps:.2f} FPS | Tổng {total_frames} frames | {duration:.2f}s")
        
        if save_to_disk and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        frame_index = 0
        saved_count = 0
        start_time = time.time()

        # Đọc tuần tự từng frame từ video
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                time_sec = frame_index / video_fps
                
                # Lưu ra đĩa nếu cờ save_to_disk được bật
                if save_to_disk and output_dir:
                    out_path = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
                    success = cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not success:
                        logging.warning(f"Lỗi không thể lưu frame {frame_index}")
                        continue
                    saved_count += 1

                # Yield frame để bước Inference phân tích ngay lập tức
                yield {
                    "frame_index": frame_index,
                    "time_sec": round(time_sec, 4),
                    "frame_data": frame # Dữ liệu ảnh gốc
                }
        finally:
            cap.release()
            elapsed = time.time() - start_time
            logging.info(f"Hoàn tất! Đã đọc {frame_index} frames. Thời gian xử lý: {elapsed:.2f}s")
            if save_to_disk:
                logging.info(f"Đã lưu vật lý {saved_count} frames vào ổ cứng.")