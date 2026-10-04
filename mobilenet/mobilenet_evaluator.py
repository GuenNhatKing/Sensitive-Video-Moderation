import logging
import json
import os
from pathlib import Path

class MobileNetEvaluator:
    def __init__(self):
        logging.info("Khởi tạo MobileNet Evaluator")

    def evaluate(self, inference_results, video_path, output_json_path=None, save_to_disk=False):
        video_abs_path = str(Path(video_path).resolve())
        
        if not inference_results:
            logging.info("Không có dữ liệu inference. Video: NORMAL.")
            result_dict = {
                "video": video_abs_path,
                "final_decision": "normal"
            }
            if save_to_disk and output_json_path:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
            return result_dict

        # Đếm số lượng frame adult và normal
        adult_count = sum(1 for r in inference_results if r["prediction"] == "adult")
        normal_count = len(inference_results) - adult_count
        total_frames = len(inference_results)

        # Đánh giá video: Xác định là Adult nếu tồn tại frame nhạy cảm
        decision = "adult" if adult_count > 0 else "normal"

        result_dict = {
            "video": video_abs_path,
            "final_decision": decision,
            "rule": "If at least ONE frame is Adult -> Video = Adult",
            "total_frames_analyzed": total_frames,
            "adult_frames": adult_count,
            "normal_frames": normal_count,
            "adult_ratio": round((adult_count / total_frames) * 100, 2) if total_frames else 0,
            "frame_results": inference_results
        }

        logging.info(f"KẾT QUẢ CUỐI CÙNG (MOBILENET): {decision.upper()} | Phát hiện {adult_count}/{total_frames} frame nhạy cảm ({result_dict['adult_ratio']}%)")

        if save_to_disk and output_json_path:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logging.info(f"Đã lưu báo cáo đánh giá tại: {output_json_path}")

        return result_dict