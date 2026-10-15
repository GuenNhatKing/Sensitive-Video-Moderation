import cv2
import logging
import os

class DenseSampler:
    def __init__(self, expand_sec=1.0, sampling_interval_sec=0.10, jpeg_quality=95):
        self.expand_sec = expand_sec
        self.sampling_interval_sec = sampling_interval_sec
        self.jpeg_quality = jpeg_quality
        
        logging.info(f"Khởi tạo Dense Sampler (Expand: ±{self.expand_sec}s, Interval: {self.sampling_interval_sec}s)")

    def _generate_timestamps(self, start_time, end_time):
        """Sinh ra các mốc thời gian cách nhau sampling_interval_sec."""
        if end_time < start_time:
            return []
        
        timestamps = []
        current = start_time
        while current <= end_time + 1e-9:
            timestamps.append(current)
            current += self.sampling_interval_sec

        if not timestamps:
            timestamps.append(start_time)
        elif abs(timestamps[-1] - end_time) > 1e-6:
            timestamps.append(end_time)
            
        return timestamps

    def sample_regions(self, video_path, regions, shots_info, previous_results, output_dir=None, save_to_disk=True):
        """
        Lấy mẫu dày đặc từ các regions.
        - previous_results: Kết quả từ bước 04 để biết frame nào ĐÃ ĐƯỢC PHÂN TÍCH.
        """
        if not regions:
            return []

        # 1. Tạo bản đồ shot và tập hợp các frame đã phân tích ở Bước 04
        shot_map = {shot["shot_id"]: shot for shot in shots_info}
        
        # Dùng set() để tra cứu siêu tốc (O(1)) xem frame đã xử lý chưa
        analyzed_frames = {res["frame_index"] for res in previous_results}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        
        output_regions = []
        total_dense_frames = 0
        global_sample_id = 0

        for region in regions:
            region_id = region["region_id"]
            shot_id = region["shot_id"]

            if shot_id not in shot_map:
                continue

            shot = shot_map[shot_id]
            shot_start_sec = float(shot["start_time_sec"])
            shot_end_sec = float(shot["end_time_sec"])
            
            # Khung giới hạn bằng frame (để loại bỏ 2 frame ngoài cùng)
            shot_start_frame = int(shot["start_frame"])
            shot_end_frame = int(shot["end_frame"])

            # Mở rộng vùng và kẹp (clamp) thời gian
            expanded_start = max(region["start_time_sec"] - self.expand_sec, shot_start_sec)
            expanded_end = min(region["end_time_sec"] + self.expand_sec, shot_end_sec, duration_sec)

            timestamps = self._generate_timestamps(expanded_start, expanded_end)
            
            region_dir = None
            if save_to_disk and output_dir:
                region_dir = os.path.join(output_dir, f"region_{region_id:03d}")
                os.makedirs(region_dir, exist_ok=True)

            dense_frames = []
            
            for local_sample_id, target_time in enumerate(timestamps, start=1):
                # Tính toán index của frame sẽ lấy
                frame_index = max(0, min(int(round(target_time * fps)), frame_count - 1))
                
                # BỘ LỌC 1: Bỏ qua nếu là 2 frame ngoài cùng của cảnh (tránh biên chuyển cảnh)
                if frame_index <= shot_start_frame or frame_index >= shot_end_frame:
                    continue
                    
                # BỘ LỌC 2: Bỏ qua nếu frame đã được phân tích ở bước 4 hoặc đã lấy trước đó
                if frame_index in analyzed_frames:
                    continue

                # Đánh dấu frame này đã được lấy để tránh trùng lặp nội bộ (trường hợp target_time gần nhau)
                analyzed_frames.add(frame_index)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()
                if not success:
                    continue
                
                actual_time = frame_index / fps
                global_sample_id += 1
                
                image_path = None
                if save_to_disk and region_dir:
                    filename = f"dense_{local_sample_id:03d}_frame_{frame_index:05d}_time_{actual_time:.2f}s.jpg"
                    image_path = os.path.join(region_dir, filename)
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

                dense_frames.append({
                    "sample_id": global_sample_id,
                    "local_sample_id": local_sample_id,
                    "frame_index": frame_index,
                    "time_sec": round(actual_time, 4),
                    "image_path": image_path,
                    "frame_data": frame if not save_to_disk else None,
                    "shot_id": shot_id
                })

            total_dense_frames += len(dense_frames)
            
            output_regions.append({
                "region_id": region_id,
                "shot_id": shot_id,
                "dense_frame_count": len(dense_frames),
                "frames": dense_frames
            })
            
            logging.debug(f"Region {region_id:02d} | Trích xuất thành công {len(dense_frames)} frames MỚI.")

        cap.release()
        logging.info(f"Hoàn tất Dense Sampling. Thu thập được {total_dense_frames} frames.")
        return output_regions