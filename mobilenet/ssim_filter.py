import cv2
import time
import logging
import os
from skimage.metrics import structural_similarity

class SSIMFilter:
    def __init__(self, ssim_threshold=0.95, target_size=(224, 224)):
        self.ssim_threshold = ssim_threshold
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        elif target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None

        size_str = f"{self.target_size[0]}x{self.target_size[1]}" if self.target_size else "Gốc"
        logging.info(f"Khởi tạo Bộ lọc SSIM (Ngưỡng loại bỏ: {self.ssim_threshold}, Kích thước lọc SSIM: {size_str})")

    def _resize_frame(self, frame, target_size=None):
        """Bộ lọc thu nhỏ ảnh về kích thước mục tiêu (mặc định 224 x 224) để tối ưu tốc độ SSIM."""
        size = target_size or self.target_size
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    def _rgb_to_gray(self, frame_bgr):
        """Chuyển đổi ảnh màu OpenCV (BGR) sang ảnh xám (Grayscale) để tính SSIM."""
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _preprocess_frame(self, frame_bgr):
        """
        Bộ lọc tiền xử lý khung hình trước khi tính SSIM:
        1. Chuyển sang ảnh xám (Grayscale) để chỉ xử lý trên 1 kênh.
        2. Thu nhỏ ảnh về 224 x 224 (target_size) để tăng tốc độ tính toán SSIM lên nhiều lần.
        """
        gray = self._rgb_to_gray(frame_bgr)
        if self.target_size:
            return self._resize_frame(gray)
        return gray

    def filter_stream(self, frame_generator, output_dir=None, save_to_disk=False):
        """
        Nhận luồng dữ liệu (generator) từ bước trích xuất, 
        lọc SSIM trực tiếp trên RAM, và yield các frame hợp lệ.
        """
        previous_gray_frame = None
        total_frames = 0
        kept_count = 0
        removed_count = 0
        start_time = time.time()

        if save_to_disk and output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            for sample in frame_generator:
                total_frames += 1
                frame_index = sample["frame_index"]
                current_bgr_frame = sample["frame_data"]
                
                # Áp dụng bộ lọc thu nhỏ về 224x224 và chuyển ảnh xám trước khi tính SSIM để tối ưu tốc độ
                current_gray_frame = self._preprocess_frame(current_bgr_frame)

                # Frame đầu tiên luôn được giữ lại
                if previous_gray_frame is None:
                    is_kept = True
                else:
                    # Tính SSIM giữa Frame(t) và Frame(t-1) trên ảnh đã thu nhỏ 224x224
                    ssim_score = structural_similarity(
                        previous_gray_frame, 
                        current_gray_frame, 
                        data_range=255
                    )
                    
                    # Áp dụng bộ lọc SSIM: Nếu độ tương đồng vượt ngưỡng thì loại bỏ frame trùng lặp
                    is_kept = (ssim_score < self.ssim_threshold)

                if is_kept:
                    kept_count += 1
                    
                    # Lưu file debug nếu cần
                    if save_to_disk and output_dir:
                        out_path = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")
                        cv2.imwrite(out_path, current_bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Trả frame về cho Pipeline
                    yield sample
                else:
                    removed_count += 1

                # CRITICAL RULE: Luôn cập nhật previous_frame là frame hiện tại (không quan tâm bị loại hay giữ)
                previous_gray_frame = current_gray_frame

                # Log tiến độ
                if total_frames % 200 == 0:
                    logging.debug(f"SSIM | Đã xử lý {total_frames} frames | Giữ: {kept_count} | Bỏ: {removed_count}")
        finally:
            elapsed = time.time() - start_time
            reduction = (removed_count / total_frames * 100) if total_frames > 0 else 0
            
            logging.info("KẾT QUẢ BỘ LỌC SSIM:")
            logging.info(f"Tổng frames đầu vào: {total_frames} | Giữ lại: {kept_count} | Loại bỏ: {removed_count} (Giảm: {reduction:.2f}%) | Thời gian lọc: {elapsed:.2f}s")