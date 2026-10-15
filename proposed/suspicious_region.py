import json
import logging
import os
from collections import defaultdict

class SuspiciousRegionDetector:
    def __init__(self, threshold=0.65, max_gap_sec=1.5, high_conf_threshold=0.97):
        self.threshold = threshold
        self.max_gap_sec = max_gap_sec
        self.high_conf_threshold = high_conf_threshold
        logging.info(
            f"Khởi tạo SuspiciousRegionDetector (Threshold: {self.threshold}, "
            f"Max Gap: {self.max_gap_sec}s, High Conf: {self.high_conf_threshold})"
        )

    def _get_suspicious_frames(self, results):
        """Lọc các frames vượt ngưỡng và sắp xếp theo shot_id, time_sec."""
        suspicious = []
        for item in results:
            adult_prob = float(item["adult_probability"])
            if adult_prob >= self.threshold:
                suspicious.append({
                    "shot_id": int(item["shot_id"]),
                    "sample_id": int(item["sample_id"]),
                    "frame_index": int(item["frame_index"]),
                    "time_sec": float(item["time_sec"]),
                    "adult_probability": adult_prob,
                    "image_path": item.get("image_path", "")
                })
                
        suspicious.sort(key=lambda x: (x["shot_id"], x["time_sec"]))
        return suspicious

    def _create_region(self, region_id, shot_id, frames):
        """Tạo dictionary chứa metadata của một vùng (gắn liền với 1 shot_id duy nhất)."""
        probabilities = [frame["adult_probability"] for frame in frames]
        peak = max(probabilities)
        return {
            "region_id": region_id,
            "shot_id": shot_id,
            "start_time_sec": frames[0]["time_sec"],
            "end_time_sec": frames[-1]["time_sec"],
            "duration_sec": round(frames[-1]["time_sec"] - frames[0]["time_sec"], 4),
            "peak_probability": round(peak, 6),
            "frame_count": len(frames),
            "frames": frames,
            "is_direct_adult": bool(peak > self.high_conf_threshold)
        }

    def _group_into_regions(self, suspicious_frames):
        """Gom nhóm các frames thành các vùng (regions) dựa trên thời gian."""
        frames_by_shot = defaultdict(list)
        for frame in suspicious_frames:
            frames_by_shot[frame["shot_id"]].append(frame)

        regions = []
        for shot_id in sorted(frames_by_shot.keys()):
            frames = frames_by_shot[shot_id]
            if not frames:
                continue

            current_region = [frames[0]]
            for frame in frames[1:]:
                previous = current_region[-1]
                gap = frame["time_sec"] - previous["time_sec"]

                if gap <= self.max_gap_sec:
                    current_region.append(frame)
                else:
                    regions.append({"shot_id": shot_id, "frames": current_region})
                    current_region = [frame]
                    
            regions.append({"shot_id": shot_id, "frames": current_region})

        # Cấp phát region_id
        final_regions = []
        for region_id, region in enumerate(regions, start=1):
            final_regions.append(self._create_region(region_id, region["shot_id"], region["frames"]))

        return final_regions

    def detect_regions(self, results, output_json_path=None, save_to_disk=True):
        """Hàm chính để thực thi từ pipeline."""
        if not results:
            logging.warning("Không có kết quả inference đầu vào.")
            return []

        suspicious_frames = self._get_suspicious_frames(results)
        logging.info(f"Tổng số frames đáng ngờ: {len(suspicious_frames)} / {len(results)}")

        if not suspicious_frames:
            logging.info("Không tìm thấy vùng đáng ngờ nào.")
            regions = []
        else:
            regions = self._group_into_regions(suspicious_frames)
            logging.info(f"Phát hiện {len(regions)} vùng nghi vấn sơ bộ từ bước lấy mẫu thưa.")

        if save_to_disk and output_json_path:
            output = {
                "threshold": self.threshold,
                "max_gap_sec": self.max_gap_sec,
                "total_samples": len(results),
                "suspicious_frame_count": len(suspicious_frames),
                "region_count": len(regions),
                "regions": regions
            }
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logging.info(f"Đã lưu metadata vùng đáng ngờ vào: {output_json_path}")

        return regions